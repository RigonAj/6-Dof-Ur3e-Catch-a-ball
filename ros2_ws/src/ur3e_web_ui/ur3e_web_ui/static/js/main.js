import { api } from "./api.js";
import { Viewer3D } from "./viewer3d.js?v=calibration-tab";
import { JogPanel } from "./jog_panel.js";
import { TargetPanel } from "./target_panel.js?v=tcp-target-base-frame";
import { RolloutPanel } from "./rollout_panel.js";
import { CalibrationPanel } from "./calibration_panel.js";

const RAD_TO_DEG = 180 / Math.PI;

function showBanner(text, isWarning = false) {
  const banner = document.getElementById("banner");
  banner.textContent = text;
  banner.className = isWarning ? "warn" : "";
}

function hideBanner() {
  document.getElementById("banner").className = "hidden";
}

function setBadge(id, text, state) {
  const badge = document.getElementById(id);
  badge.textContent = text;
  badge.className = "badge" + (state === true ? " ok" : state === false ? " bad" : "");
}

function setupTabs() {
  for (const tab of document.querySelectorAll(".tab")) {
    tab.addEventListener("click", () => {
      for (const other of document.querySelectorAll(".tab")) other.classList.remove("active");
      for (const page of document.querySelectorAll(".tab-page")) page.classList.remove("active");
      tab.classList.add("active");
      document.getElementById(`tab-${tab.dataset.tab}`).classList.add("active");
    });
  }
}

function updateStatusTab(state) {
  const tbody = document.querySelector("#joint-table tbody");
  if (state.joints) {
    if (tbody.children.length !== state.joints.names.length) {
      tbody.innerHTML = state.joints.names
        .map((name) => `<tr><td>${name.replace("_joint", "")}</td>
          <td class="num" data-jr="${name}">–</td>
          <td class="num" data-jd="${name}">–</td>
          <td class="num" data-jv="${name}">–</td></tr>`)
        .join("");
    }
    state.joints.names.forEach((name, i) => {
      document.querySelector(`[data-jr="${name}"]`).textContent = state.joints.positions_rad[i].toFixed(4);
      document.querySelector(`[data-jd="${name}"]`).textContent = (state.joints.positions_rad[i] * RAD_TO_DEG).toFixed(1);
      document.querySelector(`[data-jv="${name}"]`).textContent = (state.joints.velocities_rad_s[i] || 0).toFixed(3);
    });
  }

  if (state.tcp) {
    document.getElementById("tcp-x").textContent = state.tcp.xyz_m[0].toFixed(4);
    document.getElementById("tcp-y").textContent = state.tcp.xyz_m[1].toFixed(4);
    document.getElementById("tcp-z").textContent = state.tcp.xyz_m[2].toFixed(4);
    document.getElementById("tcp-r").textContent = (state.tcp.rpy_rad[0] * RAD_TO_DEG).toFixed(1);
    document.getElementById("tcp-p").textContent = (state.tcp.rpy_rad[1] * RAD_TO_DEG).toFixed(1);
    document.getElementById("tcp-yw").textContent = (state.tcp.rpy_rad[2] * RAD_TO_DEG).toFixed(1);
  }
}

function updateBadges(state) {
  const driver = state.driver || {};
  setBadge("badge-joints", `joints: ${driver.joint_states_alive ? "live" : "none"}`, driver.joint_states_alive);
  // The action server can exist while the controller itself was deactivated
  // by the UR controller_stopper (it then rejects every goal).
  if (driver.action_server_ready && driver.controller_active === false) {
    setBadge("badge-action", "controller: inactive", false);
  } else {
    setBadge("badge-action", `controller: ${driver.action_server_ready ? "ready" : "down"}`, driver.action_server_ready);
  }
  setBadge("badge-ik", `ik: ${driver.ik_service_ready ? "ready" : "down"}`, driver.ik_service_ready);
  if (driver.program_running === null || driver.program_running === undefined) {
    setBadge("badge-program", "program: n/a", null);
  } else {
    setBadge("badge-program", `program: ${driver.program_running ? "running" : "stopped"}`, driver.program_running);
  }
  // The speed scaling topic reports a 0..1 fraction on the real robot but
  // 100.0 on fake hardware; normalize both to percent.
  let speedText = "n/a";
  let speedPercent = null;
  if (driver.speed_scaling !== null && driver.speed_scaling !== undefined) {
    speedPercent = driver.speed_scaling <= 1.5 ? driver.speed_scaling * 100 : driver.speed_scaling;
    speedText = `${speedPercent.toFixed(0)}%`;
  }
  setBadge("badge-speed", `speed: ${speedText}`, speedPercent === null ? null : speedPercent > 0);

  if (driver.program_running === false) {
    showBanner("External Control program is not running on the pendant — motion goals will not move the robot", true);
  } else if (speedPercent !== null && speedPercent <= 0) {
    showBanner("Robot speed scaling is 0% — press Play on External Control and raise the speed slider", true);
  }
}

async function boot() {
  setupTabs();
  const viewer = new Viewer3D(document.getElementById("viewer"));
  const onError = (message) => {
    showBanner(message);
    setTimeout(hideBanner, 5000);
  };
  const jogPanel = new JogPanel({ onError });
  const targetPanel = new TargetPanel({ viewer, onError });
  const rolloutPanel = new RolloutPanel({ viewer, onError });
  const calibrationPanel = new CalibrationPanel({ viewer, onError });

  try {
    const { text, source } = await api.urdf();
    viewer.loadRobot(text);
    setBadge("badge-urdf", `urdf: ${source}`, source === "topic");
  } catch (error) {
    onError(`URDF load failed: ${error.message}`);
  }

  try {
    const limits = await api.get("/api/limits");
    jogPanel.buildRows(Object.keys(limits.joints));
  } catch (error) {
    onError(`limits load failed: ${error.message}`);
  }

  rolloutPanel.load();
  calibrationPanel.load();
  connectWebSocket(viewer, jogPanel, targetPanel, rolloutPanel, calibrationPanel);
}

function connectWebSocket(viewer, jogPanel, targetPanel, rolloutPanel, calibrationPanel) {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${location.host}/ws`);
  let gotMessage = false;
  let lastDriver = null;

  socket.onmessage = (event) => {
    gotMessage = true;
    const message = JSON.parse(event.data);
    const state = message.type === "goal_event" ? { goal: message.goal } : message;
    if (message.type === "state") {
      if (state.joints) viewer.setLiveJoints(state.joints.names, state.joints.positions_rad);
      updateStatusTab(state);
      updateBadges(state);
      const driver = state.driver || {};
      const speedPercent = driver.speed_scaling === null || driver.speed_scaling === undefined
        ? null
        : (driver.speed_scaling <= 1.5 ? driver.speed_scaling * 100 : driver.speed_scaling);
      if (state.driver && driver.program_running !== false && (speedPercent === null || speedPercent > 0)) hideBanner();
    }
    jogPanel.update(state.driver ? state : { ...state, driver: lastDriver });
    targetPanel.update(state.driver ? state : { ...state, driver: lastDriver });
    rolloutPanel.update(state.driver ? state : { ...state, driver: lastDriver });
    calibrationPanel.update(state.driver ? state : { ...state, driver: lastDriver });
    if (state.driver) lastDriver = state.driver;
  };

  socket.onclose = () => {
    showBanner("connection to server lost — reconnecting…");
    setTimeout(
      () => connectWebSocket(viewer, jogPanel, targetPanel, rolloutPanel, calibrationPanel),
      gotMessage ? 1000 : 3000,
    );
  };
}

boot();
