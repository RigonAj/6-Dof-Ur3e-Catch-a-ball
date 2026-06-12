import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
import URDFLoader from "urdf-loader";

const ROS_TO_THREE_Q = new THREE.Quaternion().setFromEuler(new THREE.Euler(-Math.PI / 2, 0, 0, "XYZ"));
const THREE_TO_ROS_Q = ROS_TO_THREE_Q.clone().invert();
const BASE_TO_BASE_LINK_Q = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, Math.PI, "XYZ"));
const BASE_LINK_TO_BASE_Q = BASE_TO_BASE_LINK_Q.clone().invert();

export class Viewer3D {
  constructor(container) {
    this.container = container;
    this.robot = null;
    this.ghost = null;
    this.ghostMaterialApplied = false;
    this.preview = null; // {plan, startMs, onProgress, onDone}
    this.targetFrame = null;
    this.targetCallback = null;
    this.targetUpdateMuted = false;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x14171c);

    this.camera = new THREE.PerspectiveCamera(50, 1, 0.01, 20);
    this.camera.position.set(0.7, 0.55, 0.7);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0.25, 0);

    this.scene.add(new THREE.HemisphereLight(0xdfe7f5, 0x202830, 1.1));
    const sun = new THREE.DirectionalLight(0xffffff, 1.4);
    sun.position.set(1.5, 2.5, 1.0);
    this.scene.add(sun);

    const grid = new THREE.GridHelper(2, 20, 0x3a4252, 0x262c38);
    this.scene.add(grid);

    this.buildTargetFrame();

    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
    this.renderer.setAnimationLoop(() => this.tick());
  }

  resize() {
    const width = this.container.clientWidth || 1;
    const height = this.container.clientHeight || 1;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  loadRobot(urdfText) {
    const loader = new URDFLoader();
    loader.packages = { ur_description: "/pkg/ur_description" };

    this.robot = loader.parse(urdfText);
    this.robot.rotation.x = -Math.PI / 2; // ROS Z-up -> three.js Y-up
    this.scene.add(this.robot);

    this.ghost = loader.parse(urdfText);
    this.ghost.rotation.x = -Math.PI / 2;
    this.ghost.visible = false;
    this.scene.add(this.ghost);
  }

  buildTargetFrame() {
    const material = new THREE.MeshStandardMaterial({
      color: 0xe0a93e,
      emissive: 0x3a2500,
      roughness: 0.45,
    });
    this.targetFrame = new THREE.Group();
    this.targetFrame.position.set(0.25, 0.35, 0.15);
    this.targetFrame.visible = false;
    this.targetFrame.add(new THREE.Mesh(new THREE.SphereGeometry(0.018, 24, 12), material));

    const ringMaterial = new THREE.MeshBasicMaterial({ color: 0xe0a93e, transparent: true, opacity: 0.55 });
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.07, 0.0025, 8, 48), ringMaterial);
    ring.rotation.x = Math.PI / 2;
    this.targetFrame.add(ring);
    this.addTargetTriad();
    this.addTargetRotationRings();
    this.scene.add(this.targetFrame);

    this.targetControls = new TransformControls(this.camera, this.renderer.domElement);
    this.targetControls.setSize(0.75);
    this.targetControls.setMode("translate");
    this.targetControls.setSpace("local");
    this.targetControls.attach(this.targetFrame);
    this.targetControls.visible = false;
    this.targetControls.addEventListener("dragging-changed", (event) => {
      this.controls.enabled = !event.value;
    });
    this.targetControls.addEventListener("objectChange", () => {
      if (!this.targetCallback || this.targetUpdateMuted) return;
      this.targetCallback(this.getTargetPose());
    });
    this.scene.add(this.targetControls);
  }

  addTargetTriad() {
    const origin = new THREE.Vector3(0, 0, 0);
    const axes = [
      { label: "X", direction: new THREE.Vector3(1, 0, 0), color: 0xff4b4b, labelPosition: [0.28, 0, 0] },
      { label: "Y", direction: new THREE.Vector3(0, 1, 0), color: 0x41c97f, labelPosition: [0, 0.28, 0] },
      { label: "Z", direction: new THREE.Vector3(0, 0, 1), color: 0x4fa3ff, labelPosition: [0, 0, 0.28] },
    ];
    for (const axis of axes) {
      const arrow = new THREE.ArrowHelper(axis.direction, origin, 0.24, axis.color, 0.06, 0.035);
      this.targetFrame.add(arrow);
      const label = this.makeAxisLabel(axis.label, axis.color);
      label.position.set(...axis.labelPosition);
      this.targetFrame.add(label);
    }
  }

  addTargetRotationRings() {
    const rings = [
      { color: 0xff4b4b, rotation: [0, Math.PI / 2, 0] },
      { color: 0x41c97f, rotation: [Math.PI / 2, 0, 0] },
      { color: 0x4fa3ff, rotation: [0, 0, 0] },
    ];
    for (const ringSpec of rings) {
      const material = new THREE.MeshBasicMaterial({
        color: ringSpec.color,
        transparent: true,
        opacity: 0.35,
        depthTest: false,
      });
      const ring = new THREE.Mesh(new THREE.TorusGeometry(0.18, 0.003, 8, 72), material);
      ring.rotation.set(...ringSpec.rotation);
      this.targetFrame.add(ring);
    }
  }

  makeAxisLabel(text, color) {
    const canvas = document.createElement("canvas");
    canvas.width = 96;
    canvas.height = 96;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#14171c";
    ctx.beginPath();
    ctx.arc(48, 48, 26, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = `#${color.toString(16).padStart(6, "0")}`;
    ctx.lineWidth = 6;
    ctx.stroke();
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 38px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, 48, 50);

    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
    const sprite = new THREE.Sprite(material);
    sprite.scale.set(0.07, 0.07, 0.07);
    return sprite;
  }

  setLiveJoints(names, positions) {
    if (!this.robot) return;
    for (let i = 0; i < names.length; i++) {
      const joint = this.robot.joints[names[i]];
      if (joint) joint.setJointValue(positions[i]);
    }
  }

  applyGhostMaterial() {
    if (!this.ghost || this.ghostMaterialApplied) return;
    const material = new THREE.MeshStandardMaterial({
      color: 0x4fa3ff,
      transparent: true,
      opacity: 0.35,
      depthWrite: false,
    });
    let meshCount = 0;
    this.ghost.traverse((child) => {
      if (child.isMesh) {
        child.material = material;
        meshCount += 1;
      }
    });
    // Meshes load asynchronously; only latch once they exist.
    if (meshCount > 0) this.ghostMaterialApplied = true;
  }

  playPreview(plan, { onProgress, onDone } = {}) {
    if (!this.ghost) return;
    this.applyGhostMaterial();
    this.ghost.visible = true;
    this.preview = { plan, startMs: performance.now(), onProgress, onDone };
  }

  showGhostPlanEnd(plan) {
    if (!this.ghost || !plan || !plan.positions || plan.positions.length === 0) return;
    this.applyGhostMaterial();
    this.ghost.visible = true;
    this.preview = null;
    this.setGhostJoints(plan.joint_names, plan.positions[plan.positions.length - 1]);
  }

  stopPreview() {
    if (this.ghost) this.ghost.visible = false;
    const preview = this.preview;
    this.preview = null;
    if (preview && preview.onDone) preview.onDone();
  }

  tickPreview() {
    if (!this.preview) return;
    const { plan, startMs, onProgress } = this.preview;
    const times = plan.time_from_start_s;
    const total = times[times.length - 1];
    const t = (performance.now() - startMs) / 1000;

    if (t >= total) {
      this.setGhostJoints(plan.joint_names, plan.positions[plan.positions.length - 1]);
      if (onProgress) onProgress(1);
      this.stopPreview();
      return;
    }

    // plan.positions[i] corresponds to times[i]; first point may be at t=0.
    let segment = 0;
    while (segment < times.length - 1 && times[segment + 1] < t) segment += 1;
    const t0 = times[segment];
    const t1 = times[segment + 1];
    const alpha = t1 > t0 ? (t - t0) / (t1 - t0) : 1;
    const a = plan.positions[segment];
    const b = plan.positions[segment + 1];
    const interpolated = a.map((value, i) => value + (b[i] - value) * alpha);
    this.setGhostJoints(plan.joint_names, interpolated);
    if (onProgress) onProgress(t / total);
  }

  setGhostJoints(names, positions) {
    for (let i = 0; i < names.length; i++) {
      const joint = this.ghost.joints[names[i]];
      if (joint) joint.setJointValue(positions[i]);
    }
  }

  enableTarget(callback) {
    this.targetCallback = callback;
    if (this.targetFrame) {
      this.targetFrame.visible = true;
      this.targetControls.visible = true;
    }
  }

  setTargetMode(mode) {
    if (!this.targetControls) return;
    this.targetControls.setMode(mode === "rotate" ? "rotate" : "translate");
    this.targetControls.setSpace("local");
  }

  setTargetPose(pose) {
    if (!this.targetFrame) return;
    this.targetUpdateMuted = true;
    const [x, y, z] = pose.xyz_m;
    this.targetFrame.position.set(-x, z, y);

    const [roll, pitch, yaw] = pose.rpy_rad;
    // ROS RPY is extrinsic X-Y-Z, i.e. R = Rz(yaw)*Ry(pitch)*Rx(roll): three.js order "ZYX".
    const baseQuat = new THREE.Quaternion().setFromEuler(new THREE.Euler(roll, pitch, yaw, "ZYX"));
    const baseLinkQuat = BASE_TO_BASE_LINK_Q.clone().multiply(baseQuat);
    this.targetFrame.quaternion.copy(ROS_TO_THREE_Q.clone().multiply(baseLinkQuat));
    this.targetFrame.visible = true;
    this.targetControls.visible = true;
    this.targetUpdateMuted = false;
  }

  getTargetPose() {
    const position = this.targetFrame.position;
    const baseLinkQuat = THREE_TO_ROS_Q.clone().multiply(this.targetFrame.quaternion);
    const baseQuat = BASE_LINK_TO_BASE_Q.clone().multiply(baseLinkQuat);
    const rpy = new THREE.Euler().setFromQuaternion(baseQuat, "ZYX");
    return {
      xyz_m: [-position.x, position.z, position.y],
      rpy_rad: [rpy.x, rpy.y, rpy.z],
    };
  }

  tick() {
    this.tickPreview();
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
