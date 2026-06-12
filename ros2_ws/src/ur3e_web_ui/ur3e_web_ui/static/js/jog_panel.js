import { api } from "./api.js";

const JOG_REPEAT_MS = 250;

export class JogPanel {
  constructor({ onError }) {
    this.onError = onError;
    this.jointNames = [];
    this.holdTimer = null;
    this.rowsBuilt = false;
    this.goalKind = null;

    document.getElementById("btn-home").addEventListener("click", () => this.moveHome());
    document.getElementById("btn-cancel").addEventListener("click", () => this.cancel());

    for (const button of document.querySelectorAll("#dashboard-buttons button")) {
      button.addEventListener("click", async () => {
        try {
          const result = await api.post(`/api/dashboard/${button.dataset.dash}`);
          this.flashStatus(`${button.dataset.dash}: ${result.success ? "ok" : "failed"} ${result.message}`);
        } catch (error) {
          this.onError(error.message);
        }
      });
    }
  }

  buildRows(jointNames) {
    if (this.rowsBuilt) return;
    this.jointNames = jointNames;
    const container = document.getElementById("jog-rows");
    for (const name of jointNames) {
      const row = document.createElement("div");
      row.className = "jog-row";
      row.innerHTML = `
        <span>${name.replace("_joint", "")}</span>
        <span class="pos" data-pos="${name}">–</span>
        <button data-jog="${name}" data-dir="-1">&minus;</button>
        <button data-jog="${name}" data-dir="1">+</button>`;
      container.appendChild(row);
    }
    for (const button of container.querySelectorAll("button[data-jog]")) {
      button.addEventListener("pointerdown", (event) => {
        event.preventDefault();
        this.startHold(button.dataset.jog, parseInt(button.dataset.dir, 10));
      });
      for (const eventName of ["pointerup", "pointerleave", "pointercancel"]) {
        button.addEventListener(eventName, () => this.stopHold());
      }
    }
    this.rowsBuilt = true;
  }

  startHold(joint, direction) {
    this.stopHold();
    this.sendJog(joint, direction);
    this.holdTimer = setInterval(() => this.sendJog(joint, direction), JOG_REPEAT_MS);
  }

  stopHold() {
    if (this.holdTimer !== null) {
      clearInterval(this.holdTimer);
      this.holdTimer = null;
    }
  }

  async sendJog(joint, direction) {
    const step = parseFloat(document.getElementById("jog-step").value);
    try {
      const result = await api.post("/api/jog", { joint, direction, step_rad: step });
      if (result.clamped) this.flashStatus(`${joint} at position limit`);
    } catch (error) {
      this.stopHold();
      this.onError(error.message);
    }
  }

  async moveHome() {
    if (!window.confirm("Move the robot slowly to the home pose?")) return;
    try {
      await api.post("/api/move_home", { confirm: true });
    } catch (error) {
      this.onError(error.message);
    }
  }

  async cancel() {
    this.stopHold();
    try {
      await api.post("/api/cancel");
    } catch (error) {
      this.onError(error.message);
    }
  }

  update(state) {
    if (state.joints && this.rowsBuilt) {
      state.joints.names.forEach((name, i) => {
        const cell = document.querySelector(`[data-pos="${name}"]`);
        if (cell) cell.textContent = state.joints.positions_rad[i].toFixed(3);
      });
    }

    const goal = state.goal || { phase: "idle" };
    this.goalKind = ["sending", "active"].includes(goal.phase) ? goal.kind : null;
    const statusBox = document.getElementById("goal-status");
    statusBox.className = "goal-status " + (goal.phase === "active" ? "active" : goal.phase);
    if (goal.phase === "idle") {
      statusBox.textContent = "idle";
    } else {
      const progress = goal.total_s ? ` ${goal.elapsed_s.toFixed(1)}/${goal.total_s.toFixed(1)}s` : "";
      const error = goal.error_string && goal.error_code !== 0 ? ` — ${goal.error_string}` : "";
      statusBox.textContent = `${goal.kind}: ${goal.phase}${goal.phase === "active" ? progress : ""}${error}`;
    }

    const showDashboard = state.driver && state.driver.dashboard_available;
    document.getElementById("dashboard-title").classList.toggle("hidden", !showDashboard);
    document.getElementById("dashboard-buttons").classList.toggle("hidden", !showDashboard);

    const speedScaling = state.driver ? state.driver.speed_scaling : null;
    const speedPercent = speedScaling === null || speedScaling === undefined
      ? null
      : (speedScaling <= 1.5 ? speedScaling * 100 : speedScaling);
    const motionBlocked = state.driver && (
      state.driver.program_running === false || (speedPercent !== null && speedPercent <= 0)
    );
    const controlsDisabled = !state.driver || !state.driver.action_server_ready ||
      motionBlocked || (this.goalKind !== null && this.goalKind !== "jog");
    for (const button of document.querySelectorAll("#jog-rows button, #btn-home")) {
      button.disabled = controlsDisabled;
    }
  }

  flashStatus(text) {
    const statusBox = document.getElementById("goal-status");
    statusBox.textContent = text;
  }
}
