"""Trang demo HTML offline (docs/demo/index.html): mở bằng trình duyệt, không cần cài đặt / mạng / server.

Gom kết quả đã sinh: output/cases (ảnh + dữ liệu 3D từ tools/case_gallery.py), output/report (biểu đồ, summary.json
từ tools/stats_report.py), output/weather_eval, output/videos. Dữ liệu 3D ghi thành file .js (không dùng fetch /
ES module vì trình duyệt chặn khi mở file:// trực tiếp); three.js r147 bản script thường trong docs/demo/lib.

Chạy (sau case_gallery.py + stats_report.py):  python tools/build_demo.py
"""
import glob
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

DEMO = os.path.join(config.PROJECT_ROOT, "docs", "demo")
OUT = config.OUTPUT_DIR


def _load(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def copy_assets():
    for sub, pattern in (("cases", os.path.join(OUT, "cases", "*.png")), ("report", os.path.join(OUT, "report", "*.png")),
                         ("videos", os.path.join(OUT, "videos", "*.mp4")), ("videos", os.path.join(OUT, "videos", "*.png"))):
        dst = os.path.join(DEMO, "assets", sub)
        os.makedirs(dst, exist_ok=True)
        for f in glob.glob(pattern):
            shutil.copy2(f, dst)
    tiles = os.path.join(OUT, "cases", "camtiles")  # 6 ảnh camera / trường hợp (case_gallery.camera_tiles)
    if os.path.isdir(tiles):
        shutil.copytree(tiles, os.path.join(DEMO, "assets", "camtiles"), dirs_exist_ok=True)
    data = os.path.join(DEMO, "data")
    os.makedirs(data, exist_ok=True)
    for f in glob.glob(os.path.join(OUT, "cases", "3d", "*.json")):
        name = Path(f).stem
        if name == "cases":
            continue
        with open(f, encoding="utf-8") as fh:
            body = fh.read()
        with open(os.path.join(data, f"{name}.js"), "w", encoding="utf-8") as fh:
            fh.write(f"window.DEMO3D=window.DEMO3D||{{}};window.DEMO3D[{json.dumps(name)}]={body};")


def page_data():
    cases = _load(os.path.join(OUT, "cases", "3d", "cases.json")) or {}
    summary = _load(os.path.join(OUT, "report", "summary.json")) or {}
    weather = _load(os.path.join(OUT, "weather_eval", "report.json")) or {"rows": []}
    weather_swarm = _load(os.path.join(OUT, "weather_eval", "report_swarm.json")) or {"rows": []}
    w = {}
    for rep, key in ((weather, "real"), (weather_swarm, "swarm")):
        for r in rep["rows"]:
            w.setdefault(key, {}).setdefault(r["cls"], {})[r["group"]] = round(100 * r["recall"])
    return {"cases": cases, "summary": {k: v for k, v in summary.items() if k != "beam_points"},
            "beams": summary.get("beam_points", {}), "weather": w,
            "videos": sorted(Path(f).name for f in glob.glob(os.path.join(DEMO, "assets", "videos", "*.mp4"))),
            "charts": sorted(Path(f).name for f in glob.glob(os.path.join(DEMO, "assets", "report", "*.png")))}


HTML = r"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>N3-08C Edge case LiDAR VN</title>
<style>
:root{--bg:#f6f5f2;--card:#fff;--ink:#141413;--ink2:#55534e;--line:#e3e1db;--accent:#2a78d6;--ok:#1a9e3a;--bad:#d62728;--conf:#ff8c00;--obj:#6250d6}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{padding:20px 24px 8px;max-width:1280px;margin:auto}h1{font-size:22px;margin:0 0 4px}header p{margin:0;color:var(--ink2)}
nav{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);z-index:5}
nav div{max-width:1280px;margin:auto;padding:0 16px;display:flex;gap:4px;overflow-x:auto}
nav button{border:0;background:none;padding:12px 14px;font:inherit;color:var(--ink2);cursor:pointer;border-bottom:2px solid transparent;white-space:nowrap}
nav button.on{color:var(--ink);border-color:var(--accent);font-weight:600}
main{max-width:1280px;margin:auto;padding:16px}section{display:none}section.on{display:block}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:16px}
h2{font-size:18px;margin:0 0 10px}h3{font-size:15px;margin:14px 0 6px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.stat b{display:block;font-size:26px;line-height:1.2}.stat span{color:var(--ink2);font-size:13px}
.steps{counter-reset:s;list-style:none;padding:0;margin:0}.steps li{counter-increment:s;padding:6px 0 6px 34px;position:relative}
.steps li:before{content:counter(s);position:absolute;left:0;top:6px;width:24px;height:24px;border-radius:50%;background:var(--accent);color:#fff;text-align:center;font-size:13px;line-height:24px}
img.full{width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:#fff}
.viewer{display:grid;grid-template-columns:260px 1fr;gap:16px}@media(max-width:820px){.viewer{grid-template-columns:1fr}}
.list button{display:block;width:100%;text-align:left;border:1px solid var(--line);background:var(--card);border-radius:8px;padding:8px 10px;margin-bottom:6px;font:inherit;cursor:pointer}
.list button.on{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}.list small{color:var(--ink2);display:block}
.list h4{margin:12px 0 6px;font-size:13px;color:var(--ink2);text-transform:uppercase;letter-spacing:.04em}
.stage{position:relative;height:560px;background:#fff;border:1px solid var(--line);border-radius:10px;overflow:hidden}
.stage canvas{display:block}.stage img{width:100%;height:100%;object-fit:contain}
.ws-top{display:grid;grid-template-columns:1.7fr 1fr;gap:10px}.ws .stage{height:470px}
.bev{margin:0;background:#fff;border:1px solid var(--line);border-radius:10px;overflow:hidden;cursor:zoom-in;height:470px;display:flex;flex-direction:column}
.bev img{width:100%;flex:1;min-height:0;object-fit:contain}.bev figcaption{padding:4px 8px;font-size:12px;color:var(--ink2);border-top:1px solid var(--line)}
.cams{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:10px}
.cam{margin:0;position:relative;border:2px solid var(--line);border-radius:8px;overflow:hidden;cursor:zoom-in;background:#000}
.cam.hit{border-color:var(--obj)}.cam img{width:100%;display:block;aspect-ratio:16/9;object-fit:cover}
.cam figcaption{position:absolute;left:0;top:0;background:#000a;color:#fff;font-size:12px;padding:2px 8px;border-bottom-right-radius:6px}
.cam .badge{position:absolute;right:6px;top:4px;background:var(--obj);color:#fff;font-size:12px;border-radius:10px;padding:1px 8px}
#lightbox{position:fixed;inset:0;background:#000d;display:none;align-items:center;justify-content:center;z-index:20;cursor:zoom-out}
#lightbox img{max-width:96vw;max-height:92vh;border-radius:6px}
@media(max-width:820px){.ws-top{grid-template-columns:1fr}.cams{grid-template-columns:repeat(2,1fr)}.ws .stage,.bev{height:360px}}
.tabs{display:flex;gap:6px;margin-bottom:8px}.tabs button{border:1px solid var(--line);background:var(--card);border-radius:6px;padding:6px 12px;cursor:pointer;font:inherit}
.tabs button.on{background:var(--ink);color:#fff;border-color:var(--ink)}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:13px;color:var(--ink2);margin-top:8px}
.legend i{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:6px}
.info{margin-top:8px;font-size:14px}.info b{font-size:16px}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:right}th:first-child,td:first-child{text-align:left}
th{color:var(--ink2);font-weight:600}
.vids{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:16px}video{width:100%;border-radius:8px;border:1px solid var(--line);background:#000}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}
.note{color:var(--ink2);font-size:13px}.hint{position:absolute;right:10px;bottom:8px;font-size:12px;color:var(--ink2);background:#fffc;padding:2px 6px;border-radius:4px}
</style></head><body>
<header><h1>N3-08C — Sinh edge case LiDAR cho giao thông Việt Nam</h1>
<p>Chèn vật thể / kịch bản / thời tiết VN vào point cloud thật của nuScenes, theo đúng vật lý LiDAR, rồi đo detector pretrained (CenterPoint-PointPillar) trên mini_val. · <a href="slides.html">Slide thuyết trình</a></p></header>
<nav><div id="nav"></div></nav>
<main>
<section id="s-overview">
 <div class="stats" id="stats"></div>
 <div class="card" style="margin-top:16px"><h2>Giải pháp</h2><ol class="steps">
  <li><b>Cảnh nền thật</b>: khung LiDAR nuScenes + 9 lần quét trước (như model đọc).</li>
  <li><b>Vật thể</b>: cắt mẫu object thật (xe máy, ô tô, người) hoặc dựng hình theo kích thước công bố rồi <b>bắn đúng 32 tia</b> của Velodyne HDL-32E (xe máy chở 2-3 người / chở hàng, ba gác, xe hàng rong, chó, bò...).</li>
  <li><b>Kịch bản VN</b>: luật vị trí / hướng / tốc độ theo xe ego (bầy xe máy, cắt đầu, ngược chiều, phố chợ...), nhiều khung liên tục.</li>
  <li><b>Ghép theo vật lý</b>: mật độ theo khoảng cách, chỉ mặt nhìn thấy, che khuất 2 chiều, mặt đất cục bộ, đủ 10 sweep.</li>
  <li><b>Thời tiết</b>: mưa, sương, giông bão (mưa rất to + mặt đường ướt + bụi nước sau bánh xe).</li>
  <li><b>Đo có đối chứng</b>: cùng vật thể, cùng cách chèn, chỉ khác tình huống → khác biệt là do tình huống.</li></ol></div>
 <div class="card"><h2>Vật thể chèn trông thế nào</h2><p class="note">Điểm LiDAR của vật thể (cam) trong cảnh (xám), cạnh xe máy / ô tô thật của nuScenes.</p><img class="full" src="assets/report/vn_closeup.png" alt="cận cảnh điểm LiDAR"></div>
</section>
<section id="s-cases"><div class="viewer"><div class="list" id="caseList"></div><div id="caseWS"></div></div></section>
<section id="s-weather"><div class="card"><h2>Cùng 1 khung bầy xe máy dưới các điều kiện thời tiết</h2>
 <div class="tabs" id="wtabs"></div><div id="wWS"></div></div>
 <div class="card"><h2>Recall theo thời tiết (41 khung mini_val)</h2><div id="wtable"></div></div>
 <div class="card"><h2>4 điều kiện cạnh nhau (LiDAR nhìn từ trên)</h2><img class="full" src="assets/cases/weather_swarm.png" alt="thời tiết"></div></section>
<div id="lightbox"><img alt=""></div>
<section id="s-videos"><div class="vids" id="vids"></div></section>
<section id="s-charts"><div class="charts" id="charts"></div></section>
<section id="s-limits"><div class="card"><h2>Hạn chế</h2><ul>
 <li>Dữ liệu nhỏ: nuScenes mini, 6 xe máy có người lái để cắt mẫu, mini_val 2 scene.</li>
 <li>Xe máy dựng hình + ray-cast chưa được detector nhận như xe máy thật (64% vs 94%): kịch bản xe máy ray-cast chỉ so với đối chứng ray-cast. Ô tô ray-cast đạt (93% vs 93%).</li>
 <li>Kích thước vật thể ray-cast, tham số thời tiết là giả định có cơ sở, chưa hiệu chỉnh bằng đo thật ở VN.</li>
 <li>Bản đồ nền Singapore / Boston (đô thị): xe bò, trâu bò là vật cản bất thường, không phải bối cảnh nông thôn.</li>
 <li>"Thấy vật cản" (box bất kỳ tại chỗ) là thước đo dễ dãi với vật thể không có lớp nuScenes.</li>
 <li>Fine-tune (T7) chưa thành công: model kém đi trên dữ liệu thật; là hướng phát triển tiếp.</li></ul></div></section>
</main>
<script src="lib/three.min.js"></script><script src="lib/OrbitControls.js"></script>
<script>
const D = __DATA__;
const COL = {ok:0x1a9e3a, missed:0xd62728, merged:0xd62728, confused:0xff8c00, pred:0x2a78d6};
const VI = {car:"ô tô",truck:"xe tải",bicycle:"xe đạp",pedestrian:"người",barrier:"rào",motorcycle:"xe máy",traffic_cone:"cọc",bus:"xe buýt",construction_vehicle:"xe công trình",trailer:"rơ-moóc"};
const pct = v => v==null||isNaN(v) ? "–" : Math.round(100*v)+"%";

// ---- nav ----
const SECTIONS=[["overview","Tổng quan"],["cases","Từng trường hợp"],["weather","Thời tiết"],["videos","Video"],["charts","Biểu đồ"],["limits","Hạn chế"]];
const nav=document.getElementById("nav");
SECTIONS.forEach(([id,label],i)=>{const b=document.createElement("button");b.textContent=label;b.onclick=()=>show(id);b.dataset.id=id;nav.appendChild(b);});
function show(id){document.querySelectorAll("section").forEach(s=>s.classList.toggle("on",s.id==="s-"+id));
 nav.querySelectorAll("button").forEach(b=>b.classList.toggle("on",b.dataset.id===id));
 if(id==="cases"&&!caseWS.current)selectCase(Object.keys(D.cases).find(k=>!k.startsWith("weather")));
 if(id==="weather"&&!wWS.current)selectWeather("weather_clear");
 window.dispatchEvent(new Event("resize"));}

// ---- tổng quan ----
const S=D.summary, m=(S.t6_motorcycle||[]), g=k=>m.find(r=>r.group===k)||{};
const cargo=(S.vn_moto_vs_rc_control||{}).cargo_moto, diff=S.t6_swarm_minus_control;
const W=D.weather.real||{};
const stats=[
 ["0%","khung nuScenes mini có bầy ≥ 5 xe máy (tối đa 2 xe)"],
 [pct(g("real").recall),"xe máy THẬT được detector tìm ra (ô tô 93%)"],
 [diff?Math.round(100*diff.diff)+" điểm":"–","bầy xe máy so với đối chứng (CI 95% "+(diff?diff.ci95.map(v=>Math.round(100*v)).join(" .. "):"")+")"],
 [cargo?Math.round(100*cargo.diff_vs_rc_control.diff)+" điểm":"–","xe máy chở hàng cồng kềnh (bị nhận thành ô tô / sót)"],
 [(W.pedestrian||{}).clear+"% → "+(W.pedestrian||{}).storm+"%","người đi bộ: trời trong → giông bão"],
 [((D.beams.dog||{})["32"]||{})["40"]??"0"," điểm LiDAR trên con chó ở 40 m (LiDAR 32 tia)"]];
document.getElementById("stats").innerHTML=stats.map(([v,t])=>`<div class="stat"><b>${v}</b><span>${t}</span></div>`).join("");

// ---- 3D viewer ----
function loadData(name,cb){if(window.DEMO3D&&window.DEMO3D[name])return cb(window.DEMO3D[name]);
 const s=document.createElement("script");s.src="data/"+name+".js";s.onload=()=>cb(window.DEMO3D[name]);document.body.appendChild(s);}
function boxLines(b,color,dashed,faint){
 const [w,l,h]=b.s, c=Math.cos(b.yaw), s=Math.sin(b.yaw), pts=[];
 const cs=[[1,1],[1,-1],[-1,-1],[-1,1]].map(([a,bb])=>[b.c[0]+c*a*l/2-s*bb*w/2, b.c[1]+s*a*l/2+c*bb*w/2]);
 const z0=b.c[2]-h/2, z1=b.c[2]+h/2;
 for(let i=0;i<4;i++){const p=cs[i],q=cs[(i+1)%4];
  pts.push(p[0],p[1],z0,q[0],q[1],z0, p[0],p[1],z1,q[0],q[1],z1, p[0],p[1],z0,p[0],p[1],z1);}
 const geo=new THREE.BufferGeometry();geo.setAttribute("position",new THREE.Float32BufferAttribute(pts,3));
 const mat=dashed?new THREE.LineDashedMaterial({color,dashSize:0.3,gapSize:0.2,transparent:true,opacity:0.7})
  :new THREE.LineBasicMaterial({color,transparent:!!faint,opacity:faint?0.25:1});
 const ls=new THREE.LineSegments(geo,mat); if(dashed)ls.computeLineDistances(); return ls;}
function render3d(st,data){
 st.stage.innerHTML="";const W0=st.stage.clientWidth,H0=st.stage.clientHeight;
 const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(window.devicePixelRatio);renderer.setSize(W0,H0);renderer.setClearColor(0xffffff);
 st.stage.appendChild(renderer.domElement);
 const hint=document.createElement("div");hint.className="hint";hint.textContent="kéo: xoay · chuột phải: dời · cuộn: phóng";st.stage.appendChild(hint);
 const scene=new THREE.Scene(), cam=new THREE.PerspectiveCamera(50,W0/H0,0.1,500);cam.up.set(0,0,1);
 const ins=data.gt.filter(b=>b.inserted); let cx=0,cy=0;
 if(ins.length){cx=ins.reduce((a,b)=>a+b.c[0],0)/ins.length/2;cy=ins.reduce((a,b)=>a+b.c[1],0)/ins.length/2;}
 cam.position.set(cx-6,cy-26,22);
 const ctl=new THREE.OrbitControls(cam,renderer.domElement);ctl.target.set(cx,cy,-1);ctl.update();
 const P=data.points,n=P.length/3,pos=new Float32Array(P),col=new Float32Array(n*3);
 for(let i=0;i<n;i++){const o=data.obj[i];col[3*i]=o?0.38:0.55;col[3*i+1]=o?0.31:0.56;col[3*i+2]=o?0.84:0.6;}
 const geo=new THREE.BufferGeometry();geo.setAttribute("position",new THREE.BufferAttribute(pos,3));geo.setAttribute("color",new THREE.BufferAttribute(col,3));
 scene.add(new THREE.Points(geo,new THREE.PointsMaterial({size:0.09,vertexColors:true})));
 const big=[];for(let i=0;i<n;i++)if(data.obj[i])big.push(P[3*i],P[3*i+1],P[3*i+2]);
 const g2=new THREE.BufferGeometry();g2.setAttribute("position",new THREE.Float32BufferAttribute(big,3));
 scene.add(new THREE.Points(g2,new THREE.PointsMaterial({size:0.16,color:0x6250d6})));
 data.pred.forEach(b=>scene.add(boxLines(b,COL.pred,true)));
 data.gt.forEach(b=>{const l=boxLines(b,COL[b.status],false,!b.inserted);scene.add(l);
  if(b.inserted){const l2=boxLines({...b,s:[b.s[0]+0.08,b.s[1]+0.08,b.s[2]+0.08]},COL[b.status],false);scene.add(l2);}});
 const ego=new THREE.Mesh(new THREE.ConeGeometry(0.8,1.8,3),new THREE.MeshBasicMaterial({color:0x141413}));ego.rotation.x=Math.PI/2;ego.position.set(0,0,-1.2);scene.add(ego);
 let alive=true;st.stop=()=>{alive=false;renderer.dispose();};
 (function loop(){if(!alive)return;renderer.render(scene,cam);requestAnimationFrame(loop);})();
 st.onresize=()=>{const w=st.stage.clientWidth,h=st.stage.clientHeight;if(!w)return;renderer.setSize(w,h);cam.aspect=w/h;cam.updateProjectionMatrix();};}
const LEG=`<span><i style="background:var(--ok)"></i>phát hiện đúng</span><span><i style="background:var(--bad)"></i>bỏ sót</span><span><i style="background:var(--conf)"></i>nhận nhầm lớp</span><span><i style="background:var(--accent);height:1px;border-top:2px dashed var(--accent)"></i>prediction của model (3D)</span><span><i style="background:var(--obj);height:8px;width:8px;border-radius:50%"></i>điểm LiDAR object chèn</span><span>▲ xe ego</span><span>Box đậm = object chèn · box mờ = object thật nuScenes</span><span>Ảnh camera là ảnh thật, KHÔNG có object chèn (chỉ chèn vào LiDAR): object chèn = điểm tím + box chiếu lên</span>`;
// ---- workspace kiểu CVAT: 3D LiDAR + LiDAR nhìn từ trên + 6 camera quanh xe ----
const CAMORDER=["CAM_FRONT_LEFT","CAM_FRONT","CAM_FRONT_RIGHT","CAM_BACK_LEFT","CAM_BACK","CAM_BACK_RIGHT"];
const CAMVI={CAM_FRONT_LEFT:"Camera trước trái",CAM_FRONT:"Camera trước",CAM_FRONT_RIGHT:"Camera trước phải",CAM_BACK_LEFT:"Camera sau trái",CAM_BACK:"Camera sau",CAM_BACK_RIGHT:"Camera sau phải"};
const lb=document.getElementById("lightbox");lb.onclick=()=>{lb.style.display="none";};
function openLB(src){lb.querySelector("img").src=src;lb.style.display="flex";}
document.addEventListener("keydown",e=>{if(e.key==="Escape")lb.style.display="none";});
function makeWS(id,withBev){const el=document.getElementById(id);el.className="ws";
 el.innerHTML=`<div class="ws-top"><div class="stage"></div>${withBev?'<figure class="bev"><img alt=""><figcaption>LiDAR nhìn từ trên — bấm để phóng to</figcaption></figure>':""}</div><div class="cams"></div><div class="legend">${LEG}</div><div class="info"></div>`;
 if(!withBev)el.querySelector(".ws-top").style.gridTemplateColumns="1fr";
 const ws={stage:el.querySelector(".stage"),bev:el.querySelector(".bev img"),cams:el.querySelector(".cams"),info:el.querySelector(".info"),current:null};
 if(ws.bev)el.querySelector(".bev").onclick=()=>openLB(ws.bev.src);return ws;}
function showWS(ws,k){const v=D.cases[k]||{};if(ws.stop)ws.stop();ws.onresize=null;ws.current=k;
 ws.stage.innerHTML='<div class="hint">đang tải…</div>';loadData(k,d=>{if(ws.current===k)render3d(ws,d);});
 if(ws.bev)ws.bev.src=`assets/cases/${v.group}_${k}.png`;
 ws.cams.innerHTML=CAMORDER.map(c=>{const n=(v.cams||{})[c]||0,src=`assets/camtiles/${k}/${c}.jpg`;
  return `<figure class="cam${n?" hit":""}" data-src="${src}"><img loading="lazy" src="${src}" alt="${c}"><figcaption>${CAMVI[c]}</figcaption>${n?`<span class="badge">${n} object chèn</span>`:""}</figure>`;}).join("");
 ws.cams.querySelectorAll(".cam").forEach(f=>f.onclick=()=>openLB(f.dataset.src));}
window.addEventListener("resize",()=>[caseWS,wWS].forEach(v=>v.onresize&&v.onresize()));

// ---- từng trường hợp ----
const caseWS=makeWS("caseWS",true);
const list=document.getElementById("caseList");
[["t6","Kịch bản giao thông (xe cắt mẫu thật)"],["vn","Vật thể VN (ray-cast)"]].forEach(([grp,title])=>{
 const h=document.createElement("h4");h.textContent=title;list.appendChild(h);
 Object.entries(D.cases).filter(([k,v])=>v.group===grp).forEach(([k,v])=>{
  const b=document.createElement("button");b.dataset.k=k;b.innerHTML=`${v.label}<small>cả kịch bản: ${pct(v.recall_all)} (n=${v.n_all})</small>`;
  b.onclick=()=>selectCase(k);list.appendChild(b);});});
function selectCase(k){const v=D.cases[k];list.querySelectorAll("button").forEach(b=>b.classList.toggle("on",b.dataset.k===k));
 showWS(caseWS,k);
 caseWS.info.innerHTML=`<b>${v.label}</b><br>Cả kịch bản (mọi khung, 2-3 seed): phát hiện <b>${pct(v.recall_all)}</b> (n=${v.n_all}). Khung minh hoạ — ${v.frame}.<br><span class="note">Khung chọn cố định: nhiều object chèn nhất trong 20 m (không chọn theo kết quả). 3D: kéo để xoay, chuột phải để dời, cuộn để phóng; ảnh: bấm để phóng to.</span>`;}

// ---- thời tiết ----
const wWS=makeWS("wWS",false);const WT=[["weather_clear","Trời trong"],["weather_rain_heavy","Mưa to"],["weather_fog_dense","Sương dày"],["weather_storm","Giông / bão"]];
const wt=document.getElementById("wtabs");
WT.forEach(([k,l])=>{const b=document.createElement("button");b.textContent=l;b.dataset.k=k;b.onclick=()=>selectWeather(k);wt.appendChild(b);});
function selectWeather(k){wt.querySelectorAll("button").forEach(b=>b.classList.toggle("on",b.dataset.k===k));showWS(wWS,k);
 const v=D.cases[k]||{};wWS.info.innerHTML=`<b>${v.label||""}</b> — ${v.frame||""} (xe máy chèn trong khung này). Ảnh camera được phủ thời tiết tương ứng (minh hoạ, task_b/camera_weather.py); số đo là trên LiDAR; màu box đổi theo kết quả detector.`;}
const conds=[["clear","Trời trong"],["rain_light","Mưa nhẹ"],["rain_moderate","Mưa vừa"],["rain_heavy","Mưa to"],["fog_light","Sương nhẹ"],["fog_moderate","Sương vừa"],["fog_dense","Sương dày"],["storm","Giông / bão"]];
const rows=[["car","Ô tô",W],["pedestrian","Người đi bộ",W],["motorcycle","Xe máy (thật)",W],["motorcycle","Xe máy trong bầy",D.weather.swarm||{}]];
document.getElementById("wtable").innerHTML="<table><tr><th></th>"+conds.map(c=>`<th>${c[1]}</th>`).join("")+"</tr>"+
 rows.map(([cls,l,src])=>"<tr><td>"+l+"</td>"+conds.map(c=>`<td>${(src[cls]||{})[c[0]]??"–"}${(src[cls]||{})[c[0]]!=null?"%":""}</td>`).join("")+"</tr>").join("")+"</table>";

// ---- video, biểu đồ ----
const VCAP={"scene-0103_swarm_pretrained.mp4":"Bầy xe máy: nuScenes gốc (trái) / có bầy xe (phải)","scene-0103_cut_in_pretrained.mp4":"Xe máy cắt đầu","scene-0103_cargo_pretrained.mp4":"Xe máy chở hàng cồng kềnh","scene-0103_dog_pretrained.mp4":"Chó lao ngang trước xe","scene-0103_swarm_storm.mp4":"Bầy xe máy: trời trong (trái) / giông bão (phải)"};
document.getElementById("vids").innerHTML=D.videos.map(v=>`<div class="card"><h3>${VCAP[v]||v}</h3><video src="assets/videos/${v}" controls preload="metadata" poster="assets/videos/${v.replace(".mp4",".png")}"></video></div>`).join("")+'<p class="note">Video là 1 scene minh hoạ (scene-0103, 2 khung / giây); số kết luận là thống kê toàn mini_val ở các tab khác.</p>';
document.getElementById("charts").innerHTML=D.charts.filter(c=>c!=="vn_closeup.png").map(c=>`<div class="card"><img class="full" src="assets/report/${c}" alt="${c}"></div>`).join("");
const h0=location.hash.slice(1);show(SECTIONS.some(s=>s[0]===h0)?h0:"overview");  // vd index.html#cases
</script></body></html>
"""


def _pct(v):
    return "–" if v is None else f"{100 * v:.0f}%"


def slides_html(d) -> str:
    """Slide thuyết trình (docs/demo/slides.html): số liệu lấy từ summary / weather như trang demo."""
    s, w = d["summary"], d["weather"].get("real", {})
    ws = d["weather"].get("swarm", {}).get("motorcycle", {})
    t6 = {r["group"]: r for r in s.get("t6_motorcycle", [])}
    diff = s.get("t6_swarm_minus_control", {"diff": float("nan"), "ci95": [float("nan")] * 2})
    vn = {r["group"]: r for r in s.get("vn_objects", [])}
    cargo = s.get("vn_moto_vs_rc_control", {}).get("cargo_moto", {})
    multi = s.get("vn_moto_vs_rc_control", {}).get("multi_rider", {})
    peds = s.get("pedestrians", {}).get("recall", {})
    dog = d["beams"].get("dog", {})
    ci = lambda r: f"{100 * r['ci95'][0]:.0f}–{100 * r['ci95'][1]:.0f}%" if r else ""  # noqa: E731
    cases = d["cases"]
    W = lambda cls, c: w.get(cls, {}).get(c, "–")  # noqa: E731
    slides = [
        ("N3-08C", f"""<h1>Sinh edge case LiDAR cho giao thông Việt Nam</h1>
<p class="lead">Chèn vật thể, kịch bản và thời tiết Việt Nam vào point cloud thật của nuScenes — đúng vật lý LiDAR —
rồi chứng minh dữ liệu sinh ra phơi bày được điểm yếu mà dữ liệu gốc không cho thấy.</p>
<p class="muted">Detector: CenterPoint-PointPillar pretrained nuScenes · Đánh giá: nuScenes mini_val · Demo: <a href="index.html">index.html</a></p>"""),
        ("1. Vấn đề", f"""<h2>nuScenes (Boston / Singapore) thiếu giao thông Việt Nam</h2>
<div class="grid3"><div class="big"><b>0%</b>khung có bầy ≥ 5 xe máy (tối đa 2 xe)</div>
<div class="big"><b>0.5%</b>khung có xe cắt đầu</div>
<div class="big"><b>0</b>mẫu xe ba gác, xe máy chở hàng, xe hàng rong, xe bò</div></div>
<p>Hệ quả đo được: ngay trên dữ liệu thật, detector chỉ tìm ra <b>{_pct(t6.get('real', {}).get('recall'))}</b> xe máy
(ô tô 93%, người 95%) — lớp phương tiện phổ biến nhất ở Việt Nam lại là lớp yếu nhất.</p>
<p class="muted">Tự thu LiDAR ở Việt Nam: cần xe gắn cảm biến, gán nhãn 3D, chờ gặp đủ tình huống hiếm → đắt và chậm.</p>"""),
        ("2. Giải pháp", """<h2>Sinh dữ liệu trên nền LiDAR thật</h2><ol class="steps">
<li><b>Cảnh nền thật</b>: khung LiDAR nuScenes + 9 lần quét trước (như model đọc)</li>
<li><b>Vật thể</b>: cắt mẫu object thật, hoặc dựng hình theo kích thước công bố rồi bắn đúng 32 tia Velodyne HDL-32E</li>
<li><b>Kịch bản VN</b>: luật vị trí / hướng / tốc độ theo xe ego, nhiều khung liên tục</li>
<li><b>Ghép theo vật lý</b>: mật độ theo khoảng cách, chỉ mặt nhìn thấy, che khuất 2 chiều, mặt đất cục bộ</li>
<li><b>Thời tiết</b>: mưa, sương, giông bão (mưa rất to + mặt đường ướt + bụi nước sau bánh xe)</li>
<li><b>Đo có đối chứng</b>: cùng vật thể, cùng cách chèn, chỉ khác tình huống</li></ol>"""),
        ("2. Danh mục", """<h2>22 kịch bản đã sinh</h2><div class="cols">
<div><h3>Giao thông</h3><ul><li>Bầy 10-20 xe máy</li><li>Xe máy cắt đầu</li><li>Xe ngược chiều</li><li>Người băng đường: 1 / nhóm 3-8 / đám đông 10-20</li><li>Xe đỗ lấn làn</li></ul></div>
<div><h3>Vật thể Việt Nam</h3><ul><li>Xe máy chở 2-3 người</li><li>Xe máy chở hàng cồng kềnh</li><li>Xe ba gác, xe máy kéo thùng</li><li>Xe hàng rong</li><li>Chó lao ra, trâu bò, xe bò</li></ul></div>
<div><h3>Kết hợp + thời tiết</h3><ul><li>Giờ cao điểm</li><li>Phố chợ</li><li>Chó giữa bầy xe</li><li>Mưa / sương / giông bão × mọi kịch bản</li></ul></div></div>"""),
        ("2. Dữ liệu có thật không?", f"""<h2>Kiểm chứng dữ liệu sinh ra</h2><table>
<tr><th>Kiểm chứng</th><th>Kết quả</th></tr>
<tr><td>Mật độ điểm xe chèn so với xe thật cùng khoảng cách</td><td>0.98 lần</td></tr>
<tr><td>Detector thấy xe máy chèn (vị trí bình thường) như xe máy thật</td><td>{_pct(t6.get('control_2w', {}).get('recall'))}</td></tr>
<tr><td>Ô tô ray-cast so với ô tô thật</td><td>{_pct(vn.get('rc_control_car', {}).get('recall'))} vs 93% → đạt</td></tr>
<tr><td>Xe máy ray-cast so với xe máy thật</td><td>{_pct(vn.get('rc_control_moto', {}).get('recall'))} vs 94% → chưa đạt, chỉ so tương đối</td></tr></table>
<img src="assets/report/vn_closeup.png" class="img">"""),
        ("3. Kết quả: bầy xe máy", f"""<h2>Bầy xe máy là điểm yếu thật</h2>
<div class="row"><div><table><tr><th>Xe máy</th><th>Recall</th><th>CI 95%</th></tr>
<tr><td>Đối chứng (đi cùng chiều)</td><td>{_pct(t6.get('control_2w', {}).get('recall'))}</td><td>{ci(t6.get('control_2w'))}</td></tr>
<tr><td><b>Bầy 10-20 xe</b></td><td><b>{_pct(t6.get('moto_swarm', {}).get('recall'))}</b></td><td>{ci(t6.get('moto_swarm'))}</td></tr>
<tr><td>Cắt đầu 3-8 m</td><td>{_pct(t6.get('cut_in', {}).get('recall'))}</td><td></td></tr></table>
<p><b>{100 * diff['diff']:.0f} điểm</b> (CI {100 * diff['ci95'][0]:.0f} .. {100 * diff['ci95'][1]:.0f}) — có ý nghĩa thống kê.
Cùng số điểm LiDAR vẫn kém hơn; bầy càng đông càng kém; lỗi chính: xe máy bị nhận thành <b>ô tô</b>.</p></div>
<img src="assets/cases/cam_moto_swarm.png" class="img"></div>"""),
        ("3. Kết quả: vật thể Việt Nam", f"""<h2>Xe máy chở hàng cồng kềnh: tụt mạnh nhất</h2>
<div class="row"><div><table><tr><th>Kịch bản</th><th>Recall</th></tr>
<tr><td>Đối chứng xe máy ray-cast</td><td>{_pct(vn.get('rc_control_moto', {}).get('recall'))}</td></tr>
<tr><td>Chở 2-3 người</td><td>{_pct(multi.get('recall', {}).get('recall'))} (không khác biệt)</td></tr>
<tr><td><b>Chở hàng cồng kềnh</b></td><td><b>{_pct(cargo.get('recall', {}).get('recall'))}</b> ({100 * cargo.get('diff_vs_rc_control', {}).get('diff', float('nan')):.0f} điểm)</td></tr>
<tr><td>Xe ba gác (thấy vật cản)</td><td>{_pct(vn.get('ba_gac_slow', {}).get('recall'))}</td></tr>
<tr><td>Chó lao ra (thấy vật cản)</td><td>{_pct(vn.get('dog_dart', {}).get('recall'))}</td></tr>
<tr><td>Trâu / bò</td><td>{_pct(vn.get('cow_road', {}).get('recall'))}</td></tr></table>
<p>Khối hàng rộng 1-1.8 m làm xe máy không giống xe máy nào trong dữ liệu train → bị nhận thành ô tô / xe đạp, 21% bị sót hẳn.</p></div>
<img src="assets/cases/cam_cargo_moto.png" class="img"></div>"""),
        ("3. Kết quả: người đi bộ", f"""<h2>Đông người không làm detector yếu đi</h2>
<div class="row"><div><table><tr><th>Số người</th><th>Recall</th></tr>
{''.join(f'<tr><td>{k}</td><td>{_pct(v["recall"])}</td></tr>' for k, v in peds.items())}</table>
<p>nuScenes có rất nhiều đám người đi bộ nhưng <b>0%</b> bầy xe máy → model yếu <b>đúng chỗ dữ liệu train thiếu</b>.
Đây là bằng chứng cho pain point: cần dữ liệu đúng tình huống Việt Nam.</p></div>
<img src="assets/cases/cam_ped_crowd.png" class="img"></div>"""),
        ("3. Kết quả: thời tiết", f"""<h2>Thời tiết xấu cộng dồn với tình huống khó</h2><table>
<tr><th>Recall</th><th>Trời trong</th><th>Mưa to</th><th>Sương dày</th><th>Giông / bão</th></tr>
<tr><td>Ô tô</td><td>{W('car', 'clear')}%</td><td>{W('car', 'rain_heavy')}%</td><td>{W('car', 'fog_dense')}%</td><td>{W('car', 'storm')}%</td></tr>
<tr><td>Người đi bộ</td><td>{W('pedestrian', 'clear')}%</td><td>{W('pedestrian', 'rain_heavy')}%</td><td>{W('pedestrian', 'fog_dense')}%</td><td><b>{W('pedestrian', 'storm')}%</b></td></tr>
<tr><td>Xe máy trong bầy</td><td>{ws.get('clear', '–')}%</td><td>{ws.get('rain_heavy', '–')}%</td><td>{ws.get('fog_dense', '–')}%</td><td><b>{ws.get('storm', '–')}%</b></td></tr></table>
<img src="assets/cases/weather_swarm.png" class="img wide">"""),
        ("3. Giới hạn cảm biến", f"""<h2>LiDAR 32 tia không thấy vật nhỏ ở xa</h2>
<p>Số điểm LiDAR trên con chó (cao 0.6 m): <b>{dog.get('32', {}).get('10', '–')}</b> điểm ở 10 m, <b>{dog.get('32', {}).get('20', '–')}</b> ở 20 m,
<b>{dog.get('32', {}).get('40', '–')}</b> ở 40 m. LiDAR 128 tia: {dog.get('128', {}).get('20', '–')} điểm ở 20 m.
Đây là giới hạn phần cứng, không phải của model.</p><img src="assets/report/lidar_beams.png" class="img wide">"""),
        ("Demo", """<h2>Demo trực quan</h2><div class="cols">
<div><video src="assets/videos/scene-0103_swarm_pretrained.mp4" controls poster="assets/videos/scene-0103_swarm_pretrained.png"></video><p>Bầy xe máy</p></div>
<div><video src="assets/videos/scene-0103_cargo_pretrained.mp4" controls poster="assets/videos/scene-0103_cargo_pretrained.png"></video><p>Xe máy chở hàng</p></div>
<div><video src="assets/videos/scene-0103_swarm_storm.mp4" controls poster="assets/videos/scene-0103_swarm_storm.png"></video><p>Bầy xe trong giông bão</p></div></div>
<p>Xem 3D từng trường hợp, góc nhìn camera, thời tiết: <a href="index.html#cases">mở trang demo</a></p>"""),
        ("Kết luận", """<h2>Kết luận và hạn chế</h2><div class="cols">
<div><h3>Đã chứng minh</h3><ul><li>Sinh được edge case Việt Nam trên nền LiDAR thật, đúng vật lý cảm biến</li>
<li>Dữ liệu chèn được detector nhận như dữ liệu thật (xe cắt mẫu, ô tô ray-cast)</li>
<li>Phơi bày điểm yếu có ý nghĩa thống kê: bầy xe máy, xe chở hàng, giông bão</li>
<li>Chỉ ra giới hạn phần cứng: vật nhỏ ở xa</li></ul></div>
<div><h3>Hạn chế / tiếp theo</h3><ul><li>nuScenes mini, 6 xe máy để cắt mẫu</li>
<li>Xe máy dựng hình chưa giống thật (cần mesh 3D)</li><li>Tham số thời tiết chưa hiệu chỉnh bằng đo thật</li>
<li>Fine-tune bằng dữ liệu VN: chưa thành công, là bước tiếp theo</li></ul></div></div>"""),
    ]
    body = "".join(f'<section class="slide"><div class="tag">{t}</div>{c}</section>' for t, c in slides)
    return SLIDES.replace("__SLIDES__", body)


SLIDES = r"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>N3-08C Slides</title><style>
:root{--bg:#f6f5f2;--ink:#141413;--ink2:#55534e;--line:#e3e1db;--accent:#2a78d6}
*{box-sizing:border-box}html,body{margin:0;height:100%;background:var(--bg);color:var(--ink);font:20px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.slide{display:none;position:absolute;inset:0;padding:48px 64px 70px;overflow:auto}.slide.on{display:block}
.tag{color:var(--accent);font-weight:600;font-size:15px;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px}
h1{font-size:44px;margin:40px 0 18px;line-height:1.15}h2{font-size:32px;margin:0 0 18px}h3{font-size:20px;margin:0 0 8px}
.lead{font-size:24px;max-width:1000px}.muted{color:var(--ink2);font-size:17px}a{color:var(--accent)}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin:10px 0 20px}
.big{background:#fff;border:1px solid var(--line);border-radius:12px;padding:18px;color:var(--ink2);font-size:17px}.big b{display:block;font-size:44px;color:var(--ink)}
.steps li{margin:8px 0}.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:24px}
.row{display:grid;grid-template-columns:1fr 1.2fr;gap:28px;align-items:start}
table{border-collapse:collapse;font-size:18px;margin-bottom:14px}th,td{border-bottom:1px solid var(--line);padding:7px 12px;text-align:left}th{color:var(--ink2)}
.img{max-width:100%;max-height:62vh;border:1px solid var(--line);border-radius:8px;background:#fff}.img.wide{max-height:44vh}
video{width:100%;border-radius:8px;background:#000}
.bar{position:fixed;left:0;right:0;bottom:0;display:flex;justify-content:space-between;align-items:center;padding:10px 20px;color:var(--ink2);font-size:15px}
.bar button{border:1px solid var(--line);background:#fff;border-radius:6px;padding:6px 14px;font:inherit;cursor:pointer}
@media(max-width:900px){.row,.grid3{grid-template-columns:1fr}.slide{padding:24px 16px 70px}h1{font-size:32px}}
</style></head><body>__SLIDES__
<div class="bar"><button onclick="go(-1)">← Trước</button><span id="pg"></span><button onclick="go(1)">Sau →</button></div>
<script>
const S=[...document.querySelectorAll(".slide")];let i=Math.max(0,Math.min(S.length-1,(parseInt(location.hash.slice(1))||1)-1));
function render(){S.forEach((s,k)=>s.classList.toggle("on",k===i));document.getElementById("pg").textContent=(i+1)+" / "+S.length;history.replaceState(null,"","#"+(i+1));}
function go(d){i=Math.max(0,Math.min(S.length-1,i+d));document.querySelectorAll("video").forEach(v=>v.pause());render();}
document.addEventListener("keydown",e=>{if(["ArrowRight","PageDown"," "].includes(e.key)){e.preventDefault();go(1);}if(["ArrowLeft","PageUp"].includes(e.key))go(-1);});
render();
</script></body></html>
"""


def main():
    copy_assets()
    data = page_data()
    html = HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    with open(os.path.join(DEMO, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    with open(os.path.join(DEMO, "slides.html"), "w", encoding="utf-8") as f:
        f.write(slides_html(data))
    print("->", os.path.join(DEMO, "index.html"), os.path.join(DEMO, "slides.html"))


if __name__ == "__main__":
    main()
