import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";

const RISK = ["#2F9E6E", "#D9A400", "#E06A1B", "#C8283B"];
const RISK_NAME = ["Low", "Medium", "High", "Critical"];
const COBALT = "#2343B0";
const $ = (s) => document.querySelector(s);

// Row layout written by u3p/pipeline.py
const F = { id: 0, cls: 1, x1: 2, y1: 3, x2: 4, y2: 5, X: 6, Z: 7, Zraw: 8, risk: 9, ttc: 10, vz: 11 };

const state = { frame: 0, view: "rgb", selected: null, boxes: true, labels: true };
let meta, frames, tracks, series;

async function load() {
  try {
    const [m, f, t, c] = await Promise.all([
      fetch("data/meta.json").then((r) => r.json()),
      fetch("data/frames.json").then((r) => r.json()),
      fetch("data/tracks.json").then((r) => r.json()),
      fetch("data/cloud.bin").then((r) => r.arrayBuffer()),
    ]);
    meta = m; frames = f; tracks = t;
    return c;
  } catch (e) {
    $("#error").hidden = false;
    throw e;
  }
}

// --------------------------------------------------------------- canvas helper
function fitCanvas(canvas) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const { width, height } = canvas.getBoundingClientRect();
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: width, h: height };
}

// --------------------------------------------------------------- video + sync
const vids = { rgb: $("#vid-rgb"), depth: $("#vid-depth") };
const activeVid = () => vids[state.view];
const frameAt = (t) => Math.max(0, Math.min(frames.length - 1, Math.round(t * meta.video.fps)));

function setFrame(i, { seek = false } = {}) {
  state.frame = i;
  if (seek) {
    const t = (i + 0.5) / meta.video.fps;
    Object.values(vids).forEach((v) => { v.currentTime = t; });
  }
  $("#scrub").value = i;
  $("#clock").textContent = `${(i / meta.video.fps).toFixed(1)} s`;
  drawOverlay(); drawBEV(); drawTimeline(); updateLive();
}

function tick() {
  const v = activeVid();
  const i = frameAt(v.currentTime);
  if (i !== state.frame) setFrame(i);
  if (!v.paused) {
    if ("requestVideoFrameCallback" in v) v.requestVideoFrameCallback(tick);
    else requestAnimationFrame(tick);
  }
}

function setPlaying(play) {
  const v = activeVid();
  if (play) { if (v.ended) v.currentTime = 0; v.play(); tick(); } else v.pause();
  $("#play").setAttribute("aria-label", play ? "Pause" : "Play");
  $("#play-icon").setAttribute("d", play ? "M5 3h3.5v14H5zM11.5 3H15v14h-3.5z" : "M5 3l12 7-12 7z");
}

// --------------------------------------------------------------- overlay
function drawOverlay() {
  const { ctx, w, h } = fitCanvas($("#overlay"));
  ctx.clearRect(0, 0, w, h);
  if (!state.boxes) return;
  const sx = w / meta.video.width, sy = h / meta.video.height;
  // ego-vehicle mask (dashboard / bonnet / roof): shaded because detections there are ignored
  const em = meta.config.ego_mask;
  if (em && (em.top_frac || em.bottom_frac)) {
    ctx.fillStyle = "rgba(14, 20, 27, 0.55)";
    if (em.top_frac) ctx.fillRect(0, 0, w, h * em.top_frac);
    if (em.bottom_frac) {
      const y0 = h * (1 - em.bottom_frac);
      ctx.fillRect(0, y0, w, h - y0);
      ctx.fillStyle = "rgba(255, 255, 255, 0.7)";
      ctx.font = "500 11px Barlow, system-ui, sans-serif";
      ctx.textBaseline = "top";
      ctx.fillText("ego vehicle (masked)", 8, y0 + 6);
    }
  }
  ctx.font = "500 12px Barlow, system-ui, sans-serif";
  ctx.textBaseline = "top";
  for (const o of frames[state.frame] || []) {
    const sel = state.selected === o[F.id];
    const dim = state.selected !== null && !sel;
    const x = o[F.x1] * sx, y = o[F.y1] * sy, bw = (o[F.x2] - o[F.x1]) * sx, bh = (o[F.y2] - o[F.y1]) * sy;
    ctx.globalAlpha = dim ? 0.35 : 1;
    ctx.lineWidth = sel ? 3 : 2;
    ctx.strokeStyle = sel ? "#fff" : RISK[o[F.risk]];
    ctx.strokeRect(x, y, bw, bh);
    if (state.labels) {
      const ttc = o[F.ttc] != null && o[F.ttc] < 20 ? `  ${o[F.ttc].toFixed(1)} s` : "";
      const label = `${o[F.id]} ${meta.classes[o[F.cls]]}  ${o[F.Z].toFixed(1)} m${ttc}`;
      const tw = ctx.measureText(label).width + 10;
      const ly = y >= 18 ? y - 18 : y;
      ctx.fillStyle = RISK[o[F.risk]];
      ctx.fillRect(x - (sel ? 1.5 : 1), ly, tw, 18);
      ctx.fillStyle = o[F.risk] === 1 ? "#18222D" : "#fff";
      ctx.fillText(label, x + 4, ly + 3);
    }
  }
  ctx.globalAlpha = 1;
}

// --------------------------------------------------------------- bird's-eye view
const BEV_RANGE = 40;
function drawBEV() {
  const { ctx, w, h } = fitCanvas($("#bev"));
  ctx.clearRect(0, 0, w, h);
  const s = (h - 24) / BEV_RANGE, ox = w / 2, oy = h - 14;
  // field of view wedge
  const half = (meta.config.camera.hfov_deg / 2) * Math.PI / 180;
  ctx.fillStyle = "rgba(255,255,255,0.04)";
  ctx.beginPath(); ctx.moveTo(ox, oy);
  ctx.lineTo(ox - Math.tan(half) * BEV_RANGE * s, oy - BEV_RANGE * s);
  ctx.lineTo(ox + Math.tan(half) * BEV_RANGE * s, oy - BEV_RANGE * s);
  ctx.closePath(); ctx.fill();
  // lane corridor used by the risk model
  const lane = meta.config.risk.lane_half_width_m * s;
  ctx.fillStyle = "rgba(35,67,176,0.16)";
  ctx.fillRect(ox - lane, oy - BEV_RANGE * s, lane * 2, BEV_RANGE * s);
  // range rings
  ctx.strokeStyle = "rgba(255,255,255,0.13)"; ctx.fillStyle = "#8C9AA6";
  ctx.font = "12px Barlow, system-ui, sans-serif"; ctx.lineWidth = 1;
  for (let d = 10; d <= BEV_RANGE; d += 10) {
    ctx.beginPath(); ctx.arc(ox, oy, d * s, Math.PI, 2 * Math.PI); ctx.stroke();
    ctx.fillText(`${d} m`, ox + 4, oy - d * s + 13);
  }
  // trails of the last second
  const look = Math.round(meta.video.fps);
  for (let k = Math.max(0, state.frame - look); k < state.frame; k++) {
    for (const o of frames[k]) {
      ctx.fillStyle = "rgba(220,227,232,0.18)";
      ctx.fillRect(ox + o[F.X] * s - 1, oy - o[F.Z] * s - 1, 2, 2);
    }
  }
  // ego
  ctx.fillStyle = "#fff";
  ctx.beginPath(); ctx.moveTo(ox, oy - 12); ctx.lineTo(ox - 7, oy + 4); ctx.lineTo(ox + 7, oy + 4); ctx.closePath(); ctx.fill();
  // objects
  ctx.font = "500 12px Barlow, system-ui, sans-serif";
  for (const o of frames[state.frame] || []) {
    const px = ox + o[F.X] * s, py = oy - o[F.Z] * s;
    if (py < 0) continue;
    const sel = state.selected === o[F.id];
    ctx.fillStyle = RISK[o[F.risk]];
    ctx.beginPath(); ctx.arc(px, py, sel ? 8 : 6, 0, 2 * Math.PI); ctx.fill();
    if (sel) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.stroke(); }
    ctx.fillStyle = "#E6ECF0"; ctx.fillText(o[F.id], px + 10, py + 4);
  }
}

// --------------------------------------------------------------- risk ribbon (scrubber background)
function drawRibbon() {
  const { ctx, w, h } = fitCanvas($("#ribbon"));
  const n = frames.length, bw = w / n;
  frames.forEach((objs, i) => {
    const peak = objs.reduce((m, o) => Math.max(m, o[F.risk]), -1);
    ctx.fillStyle = peak < 0 ? "#3A4651" : RISK[peak];
    ctx.fillRect(i * bw, 0, Math.ceil(bw) + 0.5, h);
  });
}

// --------------------------------------------------------------- distance timeline
function buildSeries() {
  const m = new Map();
  frames.forEach((objs, i) => objs.forEach((o) => {
    if (!m.has(o[F.id])) m.set(o[F.id], []);
    m.get(o[F.id]).push([i, o[F.Z], o[F.Zraw]]);
  }));
  return m;
}

const TL = { l: 44, r: 14, t: 14, b: 26 };
function drawTimeline() {
  const { ctx, w, h } = fitCanvas($("#timeline"));
  ctx.clearRect(0, 0, w, h);
  const n = frames.length;
  let zmax = 0;
  series.forEach((pts) => pts.forEach((p) => { zmax = Math.max(zmax, p[1]); }));
  zmax = Math.min(60, Math.ceil(zmax / 10) * 10 || 10);
  const X = (i) => TL.l + (i / Math.max(1, n - 1)) * (w - TL.l - TL.r);
  const Y = (z) => TL.t + (1 - Math.min(z, zmax) / zmax) * (h - TL.t - TL.b);

  ctx.font = "12px Barlow, system-ui, sans-serif"; ctx.fillStyle = "#56636F"; ctx.lineWidth = 1;
  for (let z = 0; z <= zmax; z += 10) {
    ctx.strokeStyle = z === 0 ? "#9AA6B0" : "#DDE2E6";
    ctx.beginPath(); ctx.moveTo(TL.l, Y(z)); ctx.lineTo(w - TL.r, Y(z)); ctx.stroke();
    ctx.fillText(`${z} m`, 6, Y(z) + 4);
  }
  const dur = n / meta.video.fps, step = dur > 30 ? 10 : dur > 10 ? 2 : 1;
  for (let t = 0; t <= dur; t += step) ctx.fillText(`${t} s`, X(t * meta.video.fps) - 6, h - 8);

  const draw = (pts, color, width, idx = 1) => {
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
    pts.forEach((p, k) => {
      const gap = k > 0 && p[0] - pts[k - 1][0] > 3;
      (k === 0 || gap ? ctx.moveTo : ctx.lineTo).call(ctx, X(p[0]), Y(p[idx]));
    });
    ctx.stroke();
  };
  const visible = new Set((frames[state.frame] || []).map((o) => o[F.id]));
  series.forEach((pts, id) => {
    if (id !== state.selected) draw(pts, visible.has(id) ? "#4C5965" : "#B9C2C9", visible.has(id) ? 1.6 : 1.2);
  });
  if (state.selected !== null && series.has(state.selected)) {
    draw(series.get(state.selected), "rgba(35,67,176,0.35)", 1, 2); // raw depth, for comparison
    draw(series.get(state.selected), COBALT, 2.6);
    ctx.fillStyle = COBALT;
    ctx.fillText(`Track ${state.selected}: filtered (solid) vs raw depth (faint)`, TL.l + 8, TL.t + 12);
  }
  ctx.strokeStyle = "#18222D"; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(X(state.frame), TL.t - 4); ctx.lineTo(X(state.frame), h - TL.b); ctx.stroke();
}

// --------------------------------------------------------------- live readouts
function updateLive() {
  const objs = frames[state.frame] || [];
  $("#live-count").textContent = objs.length;
  const near = objs.reduce((m, o) => Math.min(m, o[F.Z]), Infinity);
  $("#live-near").textContent = Number.isFinite(near) ? `${near.toFixed(1)} m` : "–";
  const ttc = objs.reduce((m, o) => (o[F.ttc] != null ? Math.min(m, o[F.ttc]) : m), Infinity);
  $("#live-ttc").textContent = Number.isFinite(ttc) && ttc < 20 ? `${ttc.toFixed(1)} s` : "None closing";
  if (state.selected === null) return;
  const o = objs.find((r) => r[F.id] === state.selected);
  const t = tracks.find((r) => r.track_id === state.selected);
  $("#live-sel").textContent = o
    ? `${state.selected} ${meta.classes[o[F.cls]]}, ${o[F.Z].toFixed(1)} m`
    : `${state.selected} ${t ? t.cls_name : ""}, out of view`;
}

// --------------------------------------------------------------- track table
function buildTable() {
  const body = $("#tracks tbody");
  const rank = { LOW: 0, MEDIUM: 1, HIGH: 2, CRITICAL: 3 };
  const rows = [...tracks].sort((a, b) => (b.anomaly - a.anomaly) || (rank[b.peak_risk] - rank[a.peak_risk]) || (a.min_dist - b.min_dist));
  body.innerHTML = rows.map((t) => {
    const r = rank[t.peak_risk];
    const ttc = t.min_ttc >= 99 ? "–" : `${t.min_ttc.toFixed(1)} s`;
    return `<tr tabindex="0" data-id="${t.track_id}" aria-selected="false">
      <td>${t.track_id}</td><td>${t.cls_name}</td>
      <td class="num">${t.min_dist.toFixed(1)} m</td><td class="num">${ttc}</td>
      <td><span class="risk" style="--c:${RISK[r]}">${RISK_NAME[r]}</span></td>
      <td>${t.anomaly ? '<span class="flag">Unusual</span>' : '<span class="calm">Typical</span>'}</td></tr>`;
  }).join("");
  const choose = (tr) => {
    const id = Number(tr.dataset.id);
    select(state.selected === id ? null : id, true);
  };
  body.addEventListener("click", (e) => { const tr = e.target.closest("tr"); if (tr) choose(tr); });
  body.addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.key === " ") && e.target.matches("tr")) { e.preventDefault(); choose(e.target); }
  });
}

function select(id, jump = false) {
  state.selected = id;
  document.querySelectorAll("#tracks tbody tr").forEach((tr) => tr.setAttribute("aria-selected", String(Number(tr.dataset.id) === id)));
  if (id === null) $("#live-sel").textContent = "Click a track below";
  if (jump && id !== null && series.has(id)) {
    const pts = series.get(id);
    const closest = pts.reduce((a, b) => (b[1] < a[1] ? b : a));
    setPlaying(false);
    setFrame(closest[0], { seek: true });
  } else {
    drawOverlay(); drawBEV(); drawTimeline(); updateLive();
  }
}

// --------------------------------------------------------------- 3D scene
function buildScene(buf) {
  const host = $("#scene");
  const n = meta.cloud.n;
  const xyz = new Float32Array(buf, 0, n * 3);
  const rgb = new Uint8Array(buf, n * 12, n * 3);
  const ground = new Uint8Array(buf, n * 15, n);

  const pos = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {            // camera frame (x right, y down, z forward) -> three.js (y up, -z forward)
    pos[i * 3] = xyz[i * 3]; pos[i * 3 + 1] = -xyz[i * 3 + 1]; pos[i * 3 + 2] = -xyz[i * 3 + 2];
  }
  const colors = {
    photo: Float32Array.from(rgb, (v) => v / 255),
    depth: new Float32Array(n * 3),
    ground: new Float32Array(n * 3),
  };
  const near = new THREE.Color("#E06A1B"), far = new THREE.Color("#2343B0"), tmp = new THREE.Color();
  const maxZ = meta.config.depth.max_depth_m;
  for (let i = 0; i < n; i++) {
    tmp.copy(near).lerp(far, Math.min(1, Math.sqrt(xyz[i * 3 + 2] / maxZ)));
    colors.depth.set([tmp.r, tmp.g, tmp.b], i * 3);
    const g = ground[i] ? [0.55, 0.66, 0.92] : [0.42, 0.45, 0.48];
    colors.ground.set(g, i * 3);
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  const colAttr = new THREE.BufferAttribute(colors.photo.slice(), 3);
  geom.setAttribute("color", colAttr);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color("#1D2630");
  scene.add(new THREE.Points(geom, new THREE.PointsMaterial({ size: 0.07, vertexColors: true })));

  // ego camera marker + object boxes at the key frame
  const ego = new THREE.Mesh(new THREE.ConeGeometry(0.35, 0.9, 4), new THREE.MeshBasicMaterial({ color: 0xffffff }));
  ego.rotation.x = -Math.PI / 2; scene.add(ego);
  for (const o of meta.keyframe.objects) {
    const depthExtent = o.cls === "person" || o.cls === "bicycle" ? Math.max(0.6, o.w) : Math.max(1.8, o.w * 1.6);
    const box = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(o.w, o.h, depthExtent)),
      new THREE.LineBasicMaterial({ color: RISK[o.risk] }),
    );
    box.position.set(o.x, -o.y, -(o.z + depthExtent / 2));
    scene.add(box);
  }

  const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 500);
  camera.position.set(-9, 7, 6);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  host.prepend(renderer.domElement);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, -1, -14);
  controls.enableDamping = true;
  controls.update();

  const resize = () => {
    const { width, height } = host.getBoundingClientRect();
    renderer.setSize(width, height); camera.aspect = width / height; camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(host); resize();
  let visible = true;
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; }).observe(host);
  renderer.setAnimationLoop(() => { if (visible) { controls.update(); renderer.render(scene, camera); } });

  bindSeg(host.querySelector(".seg"), "color", (mode) => { colAttr.array.set(colors[mode]); colAttr.needsUpdate = true; });
  $("#scene-caption").textContent =
    `The most critical moment, at ${(meta.keyframe.index / meta.video.fps).toFixed(1)} s, rebuilt from depth. ${n.toLocaleString()} points.`;
}

// --------------------------------------------------------------- evaluation panel
function bars(el, rows, max, fmt) {
  el.innerHTML = rows.map((r) => `<div class="bar-row"><span>${r.label}</span>
    <span class="bar-track"><span class="bar-fill ${r.good ? "good" : ""}" style="width:${(100 * r.value / max).toFixed(1)}%"></span></span>
    <span class="bar-val">${fmt(r.value)}</span></div>`).join("");
}

function buildEval() {
  const s = meta.summary;
  const names = { detection_tracking: "Detect and track", depth: "Depth (ViT)", lifting_kalman_risk: "3D, Kalman, risk" };
  const lat = Object.entries(s.latency_ms).map(([k, v]) => ({ label: names[k] || k, value: v.mean_ms }));
  bars($("#latency"), lat, Math.max(...lat.map((r) => r.value)), (v) => `${v.toFixed(1)} ms`);
  const slow = lat.reduce((a, b) => (b.value > a.value ? b : a));
  $("#fps-note").textContent = `${s.pipeline_fps} frames per second end to end. ${slow.label} is the bottleneck.`;

  const a = s.ablation;
  if (a && a.distance_jitter_raw_m != null) {
    bars($("#ablation"), [
      { label: "Raw depth", value: a.distance_jitter_raw_m },
      { label: "Kalman filter", value: a.distance_jitter_kalman_m, good: true },
    ], a.distance_jitter_raw_m, (v) => `${v.toFixed(2)} m`);
    $("#ablation-note").textContent =
      `Frame-to-frame distance jitter across ${a.tracks_evaluated} tracks fell by ${a.jitter_reduction_pct}%. ` +
      `Speed noise went from ${a.velocity_std_raw_mps} to ${a.velocity_std_kalman_mps} m/s.`;
  } else {
    $("#ablation-note").textContent = "Not enough long tracks in this clip to compare.";
  }

  const g = s.ground_plane;
  $("#cam-height").textContent = `${g.camera_height_m.toFixed(2)} m`;
  const plausible = g.camera_height_m > 1.0 && g.camera_height_m < 2.2;
  $("#ground-note").textContent =
    `Camera height above the road, from a RANSAC plane fit to the point cloud. Dashcams sit about 1.2 to 1.8 m up, ` +
    (plausible ? "so the depth scale looks right." : "so this clip's depth scale is off; try a different field of view.");

  $("#run-meta").textContent =
    `Run on ${s.video}: ${s.frames_processed} frames over ${s.duration_s} s, ${s.unique_tracks} tracked objects. ` +
    `Models: YOLOv8n, ByteTrack, Depth Anything V2 Metric Outdoor (Small).`;
}

// --------------------------------------------------------------- controls
function bindSeg(group, key, onChange) {
  group.addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    group.querySelectorAll("button").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
    onChange(b.dataset[key]);
  });
}

function bindControls() {
  const scrub = $("#scrub");
  scrub.max = frames.length - 1;
  scrub.addEventListener("input", () => { setPlaying(false); setFrame(Number(scrub.value), { seek: true }); });
  $("#play").addEventListener("click", () => setPlaying(activeVid().paused));
  Object.values(vids).forEach((v) => v.addEventListener("ended", () => setPlaying(false)));
  bindSeg(document.querySelector(".controls .seg"), "view", (view) => {
    const wasPlaying = !activeVid().paused, t = activeVid().currentTime;
    activeVid().pause();
    state.view = view;
    Object.entries(vids).forEach(([k, v]) => { v.hidden = k !== view; });
    activeVid().currentTime = t;
    if (wasPlaying) setPlaying(true);
  });
  $("#show-boxes").addEventListener("change", (e) => { state.boxes = e.target.checked; drawOverlay(); });
  $("#show-labels").addEventListener("change", (e) => { state.labels = e.target.checked; drawOverlay(); });
  $("#timeline").addEventListener("click", (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    const f = Math.round(((e.clientX - r.left - TL.l) / (r.width - TL.l - TL.r)) * (frames.length - 1));
    setPlaying(false); setFrame(Math.max(0, Math.min(frames.length - 1, f)), { seek: true });
  });
  document.addEventListener("keydown", (e) => {
    if (e.target.closest("input, button, tr, a")) return;
    if (e.code === "Space") { e.preventDefault(); setPlaying(activeVid().paused); }
    if (e.key === "ArrowRight") setFrame(Math.min(frames.length - 1, state.frame + 1), { seek: true });
    if (e.key === "ArrowLeft") setFrame(Math.max(0, state.frame - 1), { seek: true });
  });
  new ResizeObserver(() => { drawRibbon(); drawOverlay(); drawBEV(); drawTimeline(); }).observe(document.body);
}

// --------------------------------------------------------------- boot
const cloud = await load();
series = buildSeries();
bindControls();
buildTable();
buildEval();
drawRibbon();
setFrame(0);
buildScene(cloud);
await document.fonts.ready;
drawOverlay(); drawBEV(); drawTimeline();
