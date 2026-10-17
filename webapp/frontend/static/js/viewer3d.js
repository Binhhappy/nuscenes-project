/**
 * Bộ xem point cloud 3D (Three.js r128). Dựa trên PointCloudViewer3D của repo nhóm (buildPhasev1.2), sửa:
 *  - trục nuScenes LIDAR_TOP: x sang phải, y phía trước, z lên (góc "lái xe" nhìn theo +y, bản gốc nhìn theo +x)
 *  - tô màu điểm object chèn, box theo kết quả detector (đúng / sót / nhầm lớp), object thật mờ, prediction nét đứt
 */
const STATUS_COLOR = {ok: 0x22c55e, missed: 0xef4444, merged: 0xef4444, confused: 0xf59e0b, inserted: 0xfbbf24, real: 0x94a3b8};

class PointCloudViewer3D {
  constructor(container) {
    this.container = typeof container === "string" ? document.getElementById(container) : container;
    this.colorMode = "z";       // z | intensity | object
    this.display = "augmented"; // augmented | original | both
    this.pointSize = 0.1;
    this.data = null;
    const w = this.container.clientWidth, h = this.container.clientHeight;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x07090e);
    this.camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 1000);
    this.camera.up.set(0, 0, 1);
    this.renderer = new THREE.WebGLRenderer({antialias: true});
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(w, h);
    this.container.appendChild(this.renderer.domElement);
    this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    const grid = new THREE.GridHelper(100, 50, 0x0e7490, 0x1e293b);
    grid.rotation.x = Math.PI / 2; grid.position.z = -1.85;
    this.scene.add(grid);
    const ego = new THREE.Mesh(new THREE.ConeGeometry(0.9, 2.0, 3), new THREE.MeshBasicMaterial({color: 0xffffff}));
    ego.position.set(0, 0, -1.0); this.scene.add(ego);  // nón: vị trí xe ego (hướng +y)
    this.group = new THREE.Group(); this.scene.add(this.group);
    this.setView("iso");
    new ResizeObserver(() => this.resize()).observe(this.container);
    const loop = () => { requestAnimationFrame(loop); this.controls.update(); this.renderer.render(this.scene, this.camera); };
    loop();
  }
  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix(); this.renderer.setSize(w, h);
  }
  setView(v, target = [0, 0, 0]) {
    const [tx, ty] = target;
    if (v === "bev") { this.camera.position.set(tx, ty - 0.01, 70); this.controls.target.set(tx, ty, 0); }
    else if (v === "driver") { this.camera.position.set(0, -3, 1.2); this.controls.target.set(0, 25, -0.5); }
    else { this.camera.position.set(tx - 22, ty - 32, 26); this.controls.target.set(tx, ty, -1); }
  }
  /** data: {augmented: flat, stride: 3|4, obj: [0/1], original?: flat, boxes: [...], preds: [...]} */
  setData(data) { this.data = data; this.rebuild(); }
  setColorMode(m) { this.colorMode = m; this.rebuild(); }
  setDisplay(m) { this.display = m; this.rebuild(); }
  setPointSize(s) { this.pointSize = s; this.group.children.forEach(o => { if (o.isPoints) o.material.size = o.userData.big ? s * 1.8 : s; }); }

  _points(flat, stride, obj, dim) {
    const n = flat.length / stride, pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
    const c = new THREE.Color(), objPos = [];
    for (let i = 0; i < n; i++) {
      const x = flat[i * stride], y = flat[i * stride + 1], z = flat[i * stride + 2];
      pos.set([x, y, z], i * 3);
      if (dim) c.setRGB(0.25, 0.55, 0.75);
      else if (this.colorMode === "intensity" && stride === 4) { const v = Math.min(Math.max(flat[i * 4 + 3] / 60, 0.08), 1); c.setRGB(v * 0.2, v, v * 0.9); }
      else if (this.colorMode === "object") c.setRGB(0.42, 0.45, 0.5);
      // thang cố định quanh sensor (mặt đường ~ -1.85 m): vài điểm ngoại lai không làm lệch cả thang màu
      else c.setHSL(0.66 - 0.66 * Math.min(Math.max((z + 2.2) / 5.2, 0), 1), 1, 0.5);
      col.set([c.r, c.g, c.b], i * 3);
      if (obj && obj[i]) objPos.push(x, y, z);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3)); g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    const pts = new THREE.Points(g, new THREE.PointsMaterial({size: this.pointSize, vertexColors: true, transparent: true, opacity: dim ? 0.5 : 0.9}));
    const out = [pts];
    if (objPos.length && !dim) {  // điểm object chèn: tím, to hơn
      const g2 = new THREE.BufferGeometry(); g2.setAttribute("position", new THREE.Float32BufferAttribute(objPos, 3));
      const p2 = new THREE.Points(g2, new THREE.PointsMaterial({size: this.pointSize * 1.8, color: 0xc084fc})); p2.userData.big = true; out.push(p2);
    }
    return out;
  }
  _box(b, color, opts = {}) {
    const [w, l, h] = b.size || b.s, ctr = b.translation || b.c;
    const geo = new THREE.EdgesGeometry(new THREE.BoxGeometry(l, w, h));
    const mat = opts.dashed ? new THREE.LineDashedMaterial({color, dashSize: 0.35, gapSize: 0.2, transparent: true, opacity: 0.8})
      : new THREE.LineBasicMaterial({color, transparent: !!opts.faint, opacity: opts.faint ? 0.3 : 1});
    const box = new THREE.LineSegments(geo, mat);
    if (opts.dashed) box.computeLineDistances();
    box.position.set(ctr[0], ctr[1], ctr[2]);
    if (b.rotation) box.quaternion.set(b.rotation[1], b.rotation[2], b.rotation[3], b.rotation[0]).normalize();
    else box.rotation.set(0, 0, b.yaw || 0);
    if (!opts.faint && !opts.dashed) {  // vạch hướng đầu xe
      const hd = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(l / 2 + 0.5, 0, 0)]),
        new THREE.LineBasicMaterial({color}));
      box.add(hd);
    }
    return box;
  }
  rebuild() {
    while (this.group.children.length) { const o = this.group.children.pop(); o.geometry && o.geometry.dispose(); o.material && o.material.dispose(); }
    const d = this.data; if (!d) return;
    const st = d.stride || 4;
    if (this.display !== "original") this._points(d.augmented, st, d.obj, false).forEach(o => this.group.add(o));
    if (d.original && this.display !== "augmented") this._points(d.original, st, null, this.display === "both").forEach(o => this.group.add(o));
    if (this.display === "original") return;
    (d.preds || []).forEach(p => this.group.add(this._box(p, 0x60a5fa, {dashed: true})));
    (d.boxes || []).forEach(b => this.group.add(this._box(b, STATUS_COLOR[b.status] || STATUS_COLOR.inserted, {faint: !b.inserted})));
  }
}
