import { api } from "./api.js";

const SETTINGS_FIELDS = [
  { key: "max_joint_velocity", id: "setting-max-joint-velocity", label: "Max velocity" },
  { key: "max_joint_acceleration", id: "setting-max-joint-acceleration", label: "Max acceleration" },
  { key: "approach_min_duration", id: "setting-approach-min-duration", label: "Approach min" },
  { key: "min_segment_duration", id: "setting-min-segment-duration", label: "Segment min" },
];

export class RolloutPanel {
  constructor({ viewer, onError }) {
    this.viewer = viewer;
    this.onError = onError;
    this.settings = null;
    this.selectedEpisode = null;
    this.currentPlan = null;
    this.driverAlive = false;
    this.programRunning = null;
    this.lastGoalPhase = null;
    this.previewRequestId = 0;

    document.getElementById("btn-settings-apply").addEventListener("click", () => this.applySettingsFromInputs());
    document.getElementById("setting-preview-approach").addEventListener("change", () => this.revalidateSelected());
    for (const button of document.querySelectorAll("[data-replay-preset]")) {
      button.addEventListener("click", () => this.applyPreset(button.dataset.replayPreset));
    }
    document.getElementById("btn-preview").addEventListener("click", () => this.preview());
    document.getElementById("btn-preview-stop").addEventListener("click", () => {
      this.previewRequestId += 1;
      this.viewer.stopPreview();
      this.togglePreviewButtons(false);
    });
    document.getElementById("btn-execute").addEventListener("click", () => this.openExecuteModal());
    document.getElementById("modal-cancel").addEventListener("click", () => this.closeModal());
    document.getElementById("modal-confirm").addEventListener("change", (event) => {
      document.getElementById("modal-go").disabled = !event.target.checked;
    });
    document.getElementById("modal-go").addEventListener("click", () => this.execute());
  }

  async load() {
    await this.loadSettings();
    await this.loadEpisodes();
  }

  async loadSettings() {
    try {
      const data = await api.get("/api/replay_settings");
      this.renderSettings(data);
    } catch (error) {
      this.onError(`replay settings: ${error.message}`);
    }
  }

  renderSettings(data) {
    this.settings = data;
    this.fillSettings(data.limits || {});
    for (const field of SETTINGS_FIELDS) {
      const input = document.getElementById(field.id);
      const bounds = (data.bounds || {})[field.key];
      if (!bounds) continue;
      input.min = bounds.min;
      input.max = bounds.max;
      input.step = bounds.step;
    }
    this.renderSettingsStatus();
  }

  fillSettings(values) {
    for (const field of SETTINGS_FIELDS) {
      const value = values[field.key];
      if (value === null || value === undefined) continue;
      document.getElementById(field.id).value = this.formatSetting(value);
    }
  }

  renderSettingsStatus(text = null) {
    const status = document.getElementById("settings-status");
    if (text) {
      status.textContent = text;
      return;
    }
    if (!this.settings) {
      status.textContent = "";
      return;
    }
    const limits = this.settings.limits;
    status.textContent =
      `active: ${limits.max_joint_velocity} rad/s, ${limits.max_joint_acceleration} rad/s^2, ` +
      `approach ${limits.approach_min_duration}s, segment ${limits.min_segment_duration}s`;
  }

  readSettingsInputs() {
    const values = {};
    for (const field of SETTINGS_FIELDS) {
      const value = Number.parseFloat(document.getElementById(field.id).value);
      if (!Number.isFinite(value)) throw new Error(`${field.label} must be a number`);
      values[field.key] = value;
    }
    return values;
  }

  async applySettingsFromInputs() {
    try {
      await this.applySettings(this.readSettingsInputs(), "settings applied");
    } catch (error) {
      this.onError(`settings: ${error.message}`);
    }
  }

  async applyPreset(name) {
    const preset = this.settings && this.settings.presets ? this.settings.presets[name] : null;
    if (!preset) return;
    try {
      await this.applySettings(preset, `${name} preset applied`);
    } catch (error) {
      this.onError(`preset: ${error.message}`);
    }
  }

  async applySettings(values, statusText) {
    const data = await api.post("/api/replay_settings", values);
    this.renderSettings(data);
    this.renderSettingsStatus(statusText);
    await this.loadEpisodes();
    await this.revalidateSelected();
  }

  async loadEpisodes() {
    try {
      const data = await api.get("/api/rollout");
      const meta = data.metadata || {};
      document.getElementById("rollout-meta").textContent =
        `${data.path} — dt=${(meta.dt_s || 0).toFixed(4)}s, scale=${meta.action_scale}`;
      const tbody = document.querySelector("#episode-table tbody");
      tbody.innerHTML = "";
      for (const episode of data.episodes) {
        const row = document.createElement("tr");
        row.innerHTML = `
          <td>${episode.index}</td>
          <td>${episode.valid === false ? "invalid" : episode.success ? "&#10003;" : "&#10007;"}</td>
          <td class="num">${episode.steps}</td>
          <td class="num">${this.formatDuration(episode.retimed_total_s)}</td>
          <td><button data-episode="${episode.index}" ${episode.valid === false ? "disabled" : ""}>Validate</button></td>`;
        tbody.appendChild(row);
      }
      for (const button of tbody.querySelectorAll("button[data-episode]")) {
        button.addEventListener("click", () => this.validate(parseInt(button.dataset.episode, 10)));
      }
    } catch (error) {
      this.onError(`rollout: ${error.message}`);
    }
  }

  async validate(episodeIndex) {
    try {
      const includeApproach = document.getElementById("setting-preview-approach").checked && this.driverAlive;
      const approach = includeApproach ? "true" : "false";
      const plan = await api.get(`/api/rollout/${episodeIndex}/plan?approach=${approach}`);
      this.selectedEpisode = episodeIndex;
      this.currentPlan = plan;
      this.renderStats(plan);
    } catch (error) {
      this.onError(`validate: ${error.message}`);
    }
  }

  async revalidateSelected() {
    if (this.selectedEpisode === null) return;
    await this.validate(this.selectedEpisode);
  }

  renderStats(plan) {
    document.getElementById("plan-stats").classList.remove("hidden");
    document.getElementById("plan-episode").textContent =
      `episode ${plan.episode_index}` + (plan.approach_included ? " (with approach)" : " (no approach)");
    const set = (id, value) => { document.getElementById(id).textContent = value.toFixed(4); };
    set("st-raw-vel", plan.raw_stats.max_velocity_rad_s);
    set("st-ret-vel", plan.retimed_stats.max_velocity_rad_s);
    set("st-raw-acc", plan.raw_stats.max_acceleration_rad_s2);
    set("st-ret-acc", plan.retimed_stats.max_acceleration_rad_s2);
    set("st-raw-dur", plan.raw_stats.total_duration_s);
    set("st-ret-dur", plan.retimed_stats.total_duration_s);
    const limitsBox = document.getElementById("plan-limits");
    const limits = this.settings ? this.settings.limits : null;
    const limitText = limits
      ? ` Limits: ${limits.max_joint_velocity} rad/s, ${limits.max_joint_acceleration} rad/s^2.`
      : "";
    limitsBox.textContent = plan.within_limits
      ? `Retimed plan is within the configured safety limits.${limitText}`
      : `WARNING: retimed plan exceeds safety limits — execution will be refused.${limitText}`;
    limitsBox.style.color = plan.within_limits ? "" : "var(--danger)";
    document.getElementById("btn-execute").disabled = !plan.within_limits;
  }

  async preview() {
    if (!this.currentPlan) return;
    const requestId = this.previewRequestId + 1;
    this.previewRequestId = requestId;
    this.togglePreviewButtons(true);
    let referencePlan = null;
    if (this.currentPlan.approach_included && this.selectedEpisode !== null) {
      try {
        referencePlan = await api.get(`/api/rollout/${this.selectedEpisode}/plan?approach=false`);
      } catch (error) {
        this.onError(`recorded replay preview: ${error.message}`);
      }
    }
    if (requestId !== this.previewRequestId) return;
    const fill = document.getElementById("progress-fill");
    document.getElementById("exec-progress").classList.remove("hidden");
    document.getElementById("progress-label").textContent = referencePlan
      ? "preview: blue = real robot plan, amber = recorded replay"
      : "preview (ghost only — robot does not move)";
    this.viewer.playPreview(this.currentPlan, {
      referencePlan,
      onProgress: (fraction) => { fill.style.width = `${(fraction * 100).toFixed(1)}%`; },
      onDone: () => {
        this.togglePreviewButtons(false);
        document.getElementById("progress-label").textContent = "preview finished";
      },
    });
  }

  togglePreviewButtons(playing) {
    document.getElementById("btn-preview").classList.toggle("hidden", playing);
    document.getElementById("btn-preview-stop").classList.toggle("hidden", !playing);
  }

  async openExecuteModal() {
    if (this.selectedEpisode === null) return;
    try {
      // Recompute with a fresh approach segment so the modal shows what will actually run.
      const plan = await api.get(`/api/rollout/${this.selectedEpisode}/plan?approach=true`);
      this.currentPlan = plan;
      const body = document.getElementById("modal-body");
      const warning = this.programRunning === false
        ? `<p style="color: var(--danger); font-weight: 700;">External Control program is NOT running on the
           pendant — the goal will be accepted but the robot will not move.</p>`
        : "";
      const limits = this.settings ? this.settings.limits : null;
      const limitsText = limits
        ? `Limits: ${limits.max_joint_velocity} rad/s, ${limits.max_joint_acceleration} rad/s&sup2;,
           approach ${limits.approach_min_duration}s, segment ${limits.min_segment_duration}s.`
        : "";
      body.innerHTML = `${warning}
        <p>Episode <b>${plan.episode_index}</b>: ${plan.positions.length} points,
        total <b>${plan.retimed_stats.total_duration_s.toFixed(1)} s</b>
        (${plan.approach_included ? "incl. live approach" : "no approach"}), max velocity
        <b>${plan.retimed_stats.max_velocity_rad_s.toFixed(3)} rad/s</b>, max acceleration
        <b>${plan.retimed_stats.max_acceleration_rad_s2.toFixed(3)} rad/s&sup2;</b>.</p>
        <p class="hint">${limitsText}</p>`;
      document.getElementById("modal-confirm").checked = false;
      document.getElementById("modal-go").disabled = true;
      document.getElementById("modal").classList.remove("hidden");
    } catch (error) {
      this.onError(`plan: ${error.message}`);
    }
  }

  closeModal() {
    document.getElementById("modal").classList.add("hidden");
  }

  async execute() {
    this.closeModal();
    try {
      await api.post(`/api/rollout/${this.selectedEpisode}/execute`, { confirm: true });
      document.getElementById("exec-progress").classList.remove("hidden");
    } catch (error) {
      this.onError(`execute: ${error.message}`);
    }
  }

  update(state) {
    this.driverAlive = !!(state.driver && state.driver.joint_states_alive);
    this.programRunning = state.driver ? state.driver.program_running : null;

    const goal = state.goal || { phase: "idle" };
    if (goal.kind === "rollout" && goal.phase !== "idle") {
      const progressBox = document.getElementById("exec-progress");
      const fill = document.getElementById("progress-fill");
      const label = document.getElementById("progress-label");
      if (goal.phase === "active" && goal.total_s > 0) {
        progressBox.classList.remove("hidden");
        fill.style.width = `${Math.min(100, (goal.elapsed_s / goal.total_s) * 100).toFixed(1)}%`;
        label.textContent = `executing episode ${goal.episode}: ${goal.elapsed_s.toFixed(1)}/${goal.total_s.toFixed(1)} s`;
        this.lastGoalPhase = goal.phase;
      } else if (
        ["succeeded", "aborted", "canceled", "rejected"].includes(goal.phase)
        && this.lastGoalPhase !== goal.phase
      ) {
        // Render terminal states once so they don't clobber preview text later.
        progressBox.classList.remove("hidden");
        if (goal.phase === "succeeded") fill.style.width = "100%";
        label.textContent = `episode ${goal.episode}: ${goal.phase}` +
          (goal.error_string && goal.error_code !== 0 ? ` — ${goal.error_string}` : "");
        this.lastGoalPhase = goal.phase;
      }
    }
  }

  formatSetting(value) {
    return Number.parseFloat(value).toFixed(3).replace(/\.?0+$/, "");
  }

  formatDuration(value) {
    return Number.isFinite(value) ? Number.parseFloat(value).toFixed(3) : "–";
  }
}
