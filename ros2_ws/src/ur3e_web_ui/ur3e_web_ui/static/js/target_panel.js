import { api } from "./api.js";

const RAD_TO_DEG = 180 / Math.PI;
const DEG_TO_RAD = Math.PI / 180;

const FIELD_IDS = {
  x: "target-x",
  y: "target-y",
  z: "target-z",
  roll: "target-roll",
  pitch: "target-pitch",
  yaw: "target-yaw",
  duration: "target-duration",
};

const DEFAULT_TARGET_POSE = {
  xyz_m: [0.25, -0.15, 0.35],
  rpy_rad: [0, 0, 0],
};

export class TargetPanel {
  constructor({ viewer, onError }) {
    this.viewer = viewer;
    this.onError = onError;
    this.currentTcp = null;
    this.lastPlan = null;
    this.lastTargetJoints = null;
    this.initialized = false;
    this.goalKind = null;
    this.driver = null;
    this.mutingInputs = false;

    this.viewer.enableTarget((pose) => this.onTargetDragged(pose));
    this.setPose(DEFAULT_TARGET_POSE, "default target; use TCP when available");

    for (const id of Object.values(FIELD_IDS)) {
      document.getElementById(id).addEventListener("input", () => this.onInputChanged());
    }
    for (const button of document.querySelectorAll("[data-target-mode]")) {
      button.addEventListener("click", () => this.setMode(button.dataset.targetMode));
    }
    document.getElementById("btn-target-current").addEventListener("click", () => this.useCurrentTcp());
    document.getElementById("btn-target-validate").addEventListener("click", () => this.validate());
    document.getElementById("btn-target-send").addEventListener("click", () => this.execute());
    this.setMode("translate");
  }

  update(state) {
    if (state.tcp) {
      this.currentTcp = state.tcp;
      if (!this.initialized) {
        this.setPose({
          xyz_m: state.tcp.xyz_m,
          rpy_rad: state.tcp.rpy_rad,
        }, "target initialized from live TCP");
        this.initialized = true;
      }
    }

    if (state.driver) this.driver = state.driver;
    const goal = state.goal || { phase: "idle", kind: null };
    this.goalKind = ["sending", "active"].includes(goal.phase) ? goal.kind : null;
    this.updateButtons();

    if (goal.kind === "tcp" && goal.phase !== "idle") {
      const status = document.getElementById("target-status");
      status.className = "goal-status " + (goal.phase === "active" ? "active" : goal.phase);
      const progress = goal.total_s ? ` ${goal.elapsed_s.toFixed(1)}/${goal.total_s.toFixed(1)}s` : "";
      const error = goal.error_string && goal.error_code !== 0 ? ` - ${goal.error_string}` : "";
      status.textContent = `tcp: ${goal.phase}${goal.phase === "active" ? progress : ""}${error}`;
    }
  }

  setMode(mode) {
    this.viewer.setTargetMode(mode);
    for (const button of document.querySelectorAll("[data-target-mode]")) {
      button.classList.toggle("active", button.dataset.targetMode === mode);
    }
  }

  useCurrentTcp() {
    if (!this.currentTcp) {
      this.setStatus("no live TCP pose", false);
      return;
    }
    this.initialized = true;
    this.setPose({
      xyz_m: this.currentTcp.xyz_m,
      rpy_rad: this.currentTcp.rpy_rad,
    }, "target set to live TCP");
  }

  setPose(pose, statusText = null) {
    this.mutingInputs = true;
    this.writeInputs(pose);
    this.viewer.setTargetPose(pose);
    this.mutingInputs = false;
    this.invalidatePlan(statusText || "target changed");
  }

  onTargetDragged(pose) {
    this.initialized = true;
    this.mutingInputs = true;
    this.writeInputs(pose);
    this.mutingInputs = false;
    this.invalidatePlan("target changed; validate IK");
  }

  onInputChanged() {
    if (this.mutingInputs) return;
    this.initialized = true;
    try {
      const pose = this.readPose();
      this.viewer.setTargetPose(pose);
      this.invalidatePlan("target changed; validate IK");
    } catch (error) {
      this.invalidatePlan(error.message);
    }
  }

  async validate() {
    try {
      const result = await api.post("/api/tcp_target/plan", this.readRequest(false));
      this.lastPlan = result.plan;
      this.lastTargetJoints = result.target_joints_rad;
      this.viewer.showGhostPlanEnd(result.plan);
      const delta = Number.isFinite(result.max_joint_delta_rad)
        ? `, max joint move ${(result.max_joint_delta_rad * RAD_TO_DEG).toFixed(0)}°`
        : "";
      this.setStatus(`IK ok: ${result.duration_s.toFixed(2)}s${delta}`, true);
      this.updateButtons();
    } catch (error) {
      this.lastPlan = null;
      this.lastTargetJoints = null;
      this.updateButtons();
      this.onError(`target: ${error.message}`);
      this.setStatus("IK failed", false);
    }
  }

  async execute() {
    if (!this.lastPlan) return;
    if (!window.confirm("Send TCP target to the real robot?")) return;
    try {
      const result = await api.post("/api/tcp_target/execute", this.readRequest(true));
      this.setStatus(`goal accepted: ${result.duration_s.toFixed(2)}s`, true);
      this.lastPlan = null;
      this.lastTargetJoints = null;
      this.updateButtons();
    } catch (error) {
      this.onError(`target execute: ${error.message}`);
      this.setStatus("goal rejected", false);
    }
  }

  readRequest(confirm) {
    const pose = this.readPose();
    return {
      xyz_m: pose.xyz_m,
      rpy_rad: pose.rpy_rad,
      duration_s: this.readNumber(FIELD_IDS.duration),
      avoid_collisions: document.getElementById("target-avoid-collisions").checked,
      confirm,
      // On execute, the backend re-solves IK; it must land on the joints the
      // ghost previewed, otherwise the goal is rejected.
      expected_joints_rad: confirm ? this.lastTargetJoints : null,
    };
  }

  readPose() {
    return {
      xyz_m: [
        this.readNumber(FIELD_IDS.x),
        this.readNumber(FIELD_IDS.y),
        this.readNumber(FIELD_IDS.z),
      ],
      rpy_rad: [
        this.readNumber(FIELD_IDS.roll) * DEG_TO_RAD,
        this.readNumber(FIELD_IDS.pitch) * DEG_TO_RAD,
        this.readNumber(FIELD_IDS.yaw) * DEG_TO_RAD,
      ],
    };
  }

  readNumber(id) {
    const value = Number.parseFloat(document.getElementById(id).value);
    if (!Number.isFinite(value)) throw new Error(`${id} must be a number`);
    return value;
  }

  writeInputs(pose) {
    const set = (id, value, digits) => {
      document.getElementById(id).value = Number.parseFloat(value).toFixed(digits);
    };
    set(FIELD_IDS.x, pose.xyz_m[0], 4);
    set(FIELD_IDS.y, pose.xyz_m[1], 4);
    set(FIELD_IDS.z, pose.xyz_m[2], 4);
    set(FIELD_IDS.roll, pose.rpy_rad[0] * RAD_TO_DEG, 1);
    set(FIELD_IDS.pitch, pose.rpy_rad[1] * RAD_TO_DEG, 1);
    set(FIELD_IDS.yaw, pose.rpy_rad[2] * RAD_TO_DEG, 1);
  }

  invalidatePlan(text) {
    this.lastPlan = null;
    this.lastTargetJoints = null;
    this.setStatus(text, null);
    this.updateButtons();
  }

  updateButtons() {
    const driver = this.driver || {};
    const speedScaling = driver.speed_scaling;
    const speedPercent = speedScaling === null || speedScaling === undefined
      ? null
      : (speedScaling <= 1.5 ? speedScaling * 100 : speedScaling);
    const motionBlocked = driver.program_running === false || (speedPercent !== null && speedPercent <= 0);
    const busy = this.goalKind !== null;
    document.getElementById("btn-target-validate").disabled =
      !driver.joint_states_alive || !driver.ik_service_ready || busy;
    document.getElementById("btn-target-send").disabled =
      !this.lastPlan || !driver.action_server_ready || !driver.ik_service_ready || motionBlocked || busy;
  }

  setStatus(text, ok) {
    const status = document.getElementById("target-status");
    status.textContent = text || "idle";
    status.className = "goal-status" + (ok === true ? " succeeded" : ok === false ? " rejected" : "");
  }
}
