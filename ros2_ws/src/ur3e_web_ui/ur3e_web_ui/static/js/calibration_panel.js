import { api } from "./api.js";

const RAD_TO_DEG = 180 / Math.PI;

export class CalibrationPanel {
  constructor({ viewer, onError }) {
    this.viewer = viewer;
    this.onError = onError;
    this.poses = [];
    this.nextIndex = 0; // cycling pointer for "Go to next pose"
    this.supportLoaded = false;

    document.getElementById("btn-calib-save").addEventListener("click", () => this.saveCurrent());
    document.getElementById("btn-calib-next").addEventListener("click", () => this.gotoPose(this.nextIndex));
    document.getElementById("calib-pose-name").addEventListener("keydown", (event) => {
      if (event.key === "Enter") this.saveCurrent();
    });
    document.getElementById("calib-show-support").addEventListener("change", (event) => {
      this.toggleSupport(event.target.checked);
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
      row.innerHTML = `
        <td>${index === this.nextIndex ? "&#9654; " : ""}${index}</td>
        <td>${pose.name}</td>
        <td class="num">${joints}</td>
        <td><button data-goto="${index}">Go</button></td>
        <td><button data-del="${index}">&#10005;</button></td>`;
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
