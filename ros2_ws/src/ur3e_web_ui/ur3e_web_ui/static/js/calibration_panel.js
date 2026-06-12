import { api } from "./api.js";

const RAD_TO_DEG = 180 / Math.PI;

export class CalibrationPanel {
  constructor({ viewer, onError }) {
    this.viewer = viewer;
    this.onError = onError;
    this.poses = [];
    this.nextIndex = 0; // cycling pointer for "Go to next pose"
    this.supportLoaded = false;
    this.cameraResult = null;

    document.getElementById("btn-calib-save").addEventListener("click", () => this.saveCurrent());
    document.getElementById("btn-calib-next").addEventListener("click", () => this.gotoPose(this.nextIndex));
    document.getElementById("calib-pose-name").addEventListener("keydown", (event) => {
      if (event.key === "Enter") this.saveCurrent();
    });
    document.getElementById("calib-show-support").addEventListener("change", (event) => {
      this.toggleSupport(event.target.checked);
    });
    document.getElementById("calib-show-camera").addEventListener("change", (event) => {
      this.toggleCamera(event.target.checked);
    });
  }

  async load() {
    try {
      const data = await api.get("/api/calibration/poses");
      this.poses = data.poses;
      if (this.nextIndex >= this.poses.length) this.nextIndex = 0;
      document.getElementById("calib-path").textContent = `file: ${data.path}`;
      this.renderTable();
    } catch (error) {
      this.onError(`calibration poses: ${error.message}`);
    }
  }

  renderTable() {
    const tbody = document.querySelector("#calib-pose-table tbody");
    tbody.innerHTML = "";
    this.poses.forEach((pose, index) => {
      const joints = pose.joints_rad.map((value) => (value * RAD_TO_DEG).toFixed(0)).join(", ");
      const row = document.createElement("tr");
      const indexCell = document.createElement("td");
      indexCell.textContent = `${index === this.nextIndex ? "> " : ""}${index}`;
      const nameCell = document.createElement("td");
      nameCell.textContent = pose.name;
      const jointsCell = document.createElement("td");
      jointsCell.className = "num";
      jointsCell.textContent = joints;
      const gotoCell = document.createElement("td");
      const gotoButton = document.createElement("button");
      gotoButton.dataset.goto = String(index);
      gotoButton.textContent = "Go";
      gotoCell.appendChild(gotoButton);
      const deleteCell = document.createElement("td");
      const deleteButton = document.createElement("button");
      deleteButton.dataset.del = String(index);
      deleteButton.textContent = "X";
      deleteCell.appendChild(deleteButton);
      row.append(indexCell, nameCell, jointsCell, gotoCell, deleteCell);
      tbody.appendChild(row);
    });
    for (const button of tbody.querySelectorAll("button[data-goto]")) {
      button.addEventListener("click", () => this.gotoPose(parseInt(button.dataset.goto, 10)));
    }
    for (const button of tbody.querySelectorAll("button[data-del]")) {
      button.addEventListener("click", () => this.deletePose(parseInt(button.dataset.del, 10)));
    }
    document.getElementById("btn-calib-next").disabled = this.poses.length === 0;
  }

  async saveCurrent() {
    const input = document.getElementById("calib-pose-name");
    try {
      const result = await api.post("/api/calibration/poses", { name: input.value || null });
      input.value = "";
      this.setStatus(`saved ${result.pose.name} (${result.count} poses)`, "succeeded");
      await this.load();
    } catch (error) {
      this.onError(`save pose: ${error.message}`);
    }
  }

  async deletePose(index) {
    const pose = this.poses[index];
    if (!pose) return;
    if (!window.confirm(`Delete pose "${pose.name}"?`)) return;
    try {
      await api.del(`/api/calibration/poses/${index}`);
      this.setStatus(`deleted ${pose.name}`, "succeeded");
      await this.load();
    } catch (error) {
      this.onError(`delete pose: ${error.message}`);
    }
  }

  async gotoPose(index) {
    try {
      // Ghost-preview the exact joint target first, then confirm the move.
      const plan = await api.get(`/api/calibration/poses/${index}/plan`);
      this.viewer.showGhostPlanEnd(plan);
      const duration = plan.time_from_start_s[plan.time_from_start_s.length - 1];
      const maxDelta = (plan.max_joint_delta_rad * RAD_TO_DEG).toFixed(0);
      const go = window.confirm(
        `Move robot to "${plan.pose.name}" (max joint move ${maxDelta}°, ${duration.toFixed(1)} s)?\n` +
        "The blue ghost shows the target configuration.",
      );
      if (!go) return;
      const result = await api.post(`/api/calibration/poses/${index}/goto`, { confirm: true });
      this.setStatus(`moving to ${result.pose.name} (${result.duration_s.toFixed(1)} s)`, "active");
      this.nextIndex = (index + 1) % this.poses.length;
      this.renderTable();
    } catch (error) {
      this.onError(`go to pose: ${error.message}`);
    }
  }

  async toggleSupport(checked) {
    const status = document.getElementById("calib-support-status");
    try {
      if (checked && !this.supportLoaded) {
        const response = await fetch("/static/models/support_mount.json", { cache: "no-store" });
        if (!response.ok) throw new Error(`support_mount.json: HTTP ${response.status}`);
        const config = await response.json();
        await this.viewer.loadToolMesh(config);
        this.supportLoaded = true;
        status.textContent =
          `${config.glb_url} attached to tool0 — adjust xyz_m/rpy_rad in static/models/support_mount.json`;
      }
      this.viewer.setToolMeshVisible(checked);
    } catch (error) {
      document.getElementById("calib-show-support").checked = false;
      status.textContent = `support load failed: ${error.message}`;
    }
  }

  async toggleCamera(checked) {
    const status = document.getElementById("calib-camera-status");
    try {
      if (checked && !this.cameraResult) {
        this.cameraResult = await api.get("/api/calibration/camera");
        const transform = this.cameraResult.T_base_camera;
        this.viewer.setCameraFrame(transform);
        const xyz = transform.xyz.map((value) => value.toFixed(3)).join(", ");
        const samples = this.cameraResult.sample_count;
        status.textContent =
          `${transform.parent} -> ${transform.child}: [${xyz}] m` +
          (samples ? ` (${samples} samples, ${this.cameraResult.path})` : ` (${this.cameraResult.path})`);
      }
      this.viewer.setCameraFrameVisible(checked);
    } catch (error) {
      document.getElementById("calib-show-camera").checked = false;
      this.cameraResult = null;
      status.textContent = `camera result: ${error.message}`;
    }
  }

  update(state) {
    const goal = state.goal || { phase: "idle", kind: null };
    if (goal.kind !== "calibration" || goal.phase === "idle") return;
    const progress = goal.total_s && goal.phase === "active"
      ? ` ${goal.elapsed_s.toFixed(1)}/${goal.total_s.toFixed(1)}s`
      : "";
    const error = goal.error_string && goal.error_code !== 0 ? ` - ${goal.error_string}` : "";
    this.setStatus(`calibration move: ${goal.phase}${progress}${error}`, goal.phase);
  }

  setStatus(text, phase = null) {
    const status = document.getElementById("calib-status");
    status.textContent = text;
    status.className = "goal-status" + (phase ? ` ${phase}` : "");
  }
}
