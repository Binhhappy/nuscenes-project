"""Đóng gói site tĩnh cho Hugging Face Static Space: slide + demo 3D offline + video giới thiệu.

Nguồn: docs/demo/ (trinh_bay.html, index.html, lib/, data/, assets/), output/videos/demo_gioi_thieu.mp4,
assets/vn_objects/moto_multi/side.png. Ra: deploy/hf_static/ (không commit, xem .gitignore).

Chạy:  python deploy/build_static.py
Đẩy:   hf upload <user>/<space> deploy/hf_static . --repo-type space
"""
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "docs" / "demo"
OUT = ROOT / "deploy" / "hf_static"
NAME = "VN-LiDAR Augment: Sinh dữ liệu LiDAR giao thông Việt Nam"
DESC = ("Công cụ sinh dữ liệu LiDAR có sẵn nhãn 3D cho xe tự lái, xây trên bộ dữ liệu nuScenes. Hệ thống chèn vật thể thật vào cảnh khác và mô phỏng mưa, sương, giông theo đúng vật lý cảm biến, tạo ra các tình huống giao thông Việt Nam mà dữ liệu gốc chưa có: bầy xe máy, xe ba gác, xe hàng rong, xe chở hàng cồng kềnh. Kết quả đạt 100% cảnh hợp lệ, khoảng 9 400 cảnh mỗi giờ trên một laptop.")
ATTR = ("Dữ liệu nuScenes © Motional, giấy phép CC BY-NC-SA 4.0 · Dự án học tập phi thương mại, không liên kết với "
        "Motional · Ảnh vật thể VN: Wikimedia Commons (nguồn và giấy phép trong SOURCES.md)")

README = """---
title: VN-LiDAR Augment
emoji: 🛵
colorFrom: indigo
colorTo: blue
sdk: static
app_file: index.html
short_description: Sinh edge case LiDAR giao thông Việt Nam từ nuScenes
license: cc-by-nc-sa-4.0
---

# {name}

{desc}

- `slides.html`: slide giới thiệu (phím ← → , F toàn màn hình)
- `demo.html`: demo 3D tương tác các kịch bản đã sinh
- `assets/videos/demo_gioi_thieu.mp4`: video giới thiệu có lồng tiếng

{attr}
"""

INDEX = """<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VN-LiDAR Augment</title><style>
:root{{--bg:#070b14;--card:#111a2e;--line:#1f2b45;--ink:#eef2fb;--ink2:#9aa7c2;--c1:#22d3ee;--c2:#a78bfa}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:radial-gradient(1000px 600px at 85% -10%,rgba(167,139,250,.15),transparent 60%),
 radial-gradient(900px 600px at -10% 110%,rgba(34,211,238,.12),transparent 60%),var(--bg);color:var(--ink);
 font-family:"Segoe UI",system-ui,-apple-system,Roboto,sans-serif;min-height:100vh;padding:56px 20px}}
.wrap{{max-width:1120px;margin:0 auto}}
.k{{color:var(--c1);font-weight:700;letter-spacing:.16em;text-transform:uppercase;font-size:14px}}
h1{{font-size:clamp(34px,6vw,64px);line-height:1.05;margin:14px 0 16px;font-weight:800}}
.g{{background:linear-gradient(90deg,var(--c1),var(--c2));-webkit-background-clip:text;background-clip:text;color:transparent}}
p.s{{color:var(--ink2);font-size:clamp(16px,2.2vw,21px);max-width:760px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px;margin:42px 0}}
a.c{{display:block;background:var(--card);border:1px solid var(--line);border-radius:20px;padding:26px;color:var(--ink);
 text-decoration:none;transition:transform .15s,border-color .15s}}
a.c:hover{{transform:translateY(-3px);border-color:var(--c1)}}
a.c b{{display:block;font-size:24px;margin:10px 0 6px}}a.c span{{color:var(--ink2);font-size:16px}}
a.c i{{font-style:normal;font-size:34px}}
video{{width:100%;border-radius:18px;border:1px solid var(--line);background:#000}}
.f{{color:#64708a;font-size:13px;margin-top:40px;line-height:1.6}}
</style></head><body><div class="wrap">
<div class="k">VN-LiDAR Augment · N3-08C</div>
<h1>Giao thông Việt Nam<br><span class="g">sinh ra từ dữ liệu thật</span></h1>
<p class="s">{desc}</p>
<div class="cards">
 <a class="c" href="slides.html"><i>🎞️</i><b>Slide giới thiệu</b><span>13 slide, 3D xoay được · phím ← →, F toàn màn hình</span></a>
 <a class="c" href="demo.html"><i>🧭</i><b>Demo 3D tương tác</b><span>23 kịch bản đã sinh · thời tiết · video · biểu đồ</span></a>
 <a class="c" href="#video"><i>▶️</i><b>Video giới thiệu</b><span>3 phút 36 giây, có lồng tiếng</span></a>
</div>
<video id="video" src="assets/videos/demo_gioi_thieu.mp4" controls preload="metadata" playsinline></video>
<p class="f">{attr}</p>
</div></body></html>
"""


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for d in ("lib", "data", "assets"):
        shutil.copytree(DEMO / d, OUT / d)
    shutil.copy2(ROOT / "output" / "videos" / "demo_gioi_thieu.mp4", OUT / "assets" / "videos" / "demo_gioi_thieu.mp4")
    shutil.copy2(ROOT / "assets" / "vn_objects" / "moto_multi" / "side.png", OUT / "assets" / "img" / "moto_multi.png")
    shutil.copy2(ROOT / "assets" / "vn_objects" / "SOURCES.md", OUT / "SOURCES.md")

    slides = (DEMO / "trinh_bay.html").read_text(encoding="utf-8")
    slides = slides.replace('src="../../assets/vn_objects/moto_multi/side.png"', 'src="assets/img/moto_multi.png"')
    slides = slides.replace('<a class="cta" href="http://localhost:8060" target="_blank">Demo trực tiếp →</a>',
                            '<a class="cta" href="demo.html">Xem demo tương tác →</a>')
    assert "localhost" not in slides and "../../" not in slides
    (OUT / "slides.html").write_text(slides, encoding="utf-8")

    demo = (DEMO / "index.html").read_text(encoding="utf-8")
    demo = re.sub(r'<section id="s-limits">.*?</section>\n?', "", demo, flags=re.S)
    demo = demo.replace(',["limits","Hạn chế"]', "")
    demo = demo.replace("</main>", f'<p style="color:#888;font-size:13px;margin:24px 0">{ATTR}</p></main>', 1)
    assert "s-limits" not in demo and "Hạn chế" not in demo
    (OUT / "demo.html").write_text(demo, encoding="utf-8")

    (OUT / "index.html").write_text(INDEX.format(attr=ATTR, desc=DESC), encoding="utf-8")
    (OUT / "README.md").write_text(README.format(attr=ATTR, name=NAME, desc=DESC), encoding="utf-8")
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"saved {OUT}  ({size / 1e6:.0f} MB, {sum(1 for f in OUT.rglob('*') if f.is_file())} file)")


if __name__ == "__main__":
    main()
