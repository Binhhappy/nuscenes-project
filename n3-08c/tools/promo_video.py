"""Video giới thiệu có lồng tiếng: trình chiếu docs/demo/trinh_bay.html rồi thao tác web app theo docs/kich_ban_demo.md.

Cần: web app đang chạy (python run_app.py, mặc định http://localhost:8060 qua --url), các gói edge-tts, playwright
(+ `python -m playwright install chromium`), imageio-ffmpeg. Lời đọc tạo bằng edge-tts (dịch vụ online của Microsoft).

Chạy: python tools/promo_video.py [--voice vi-VN-HoaiMyNeural] [--url http://localhost:8060]
Ra:   output/videos/demo_gioi_thieu.mp4
"""
import argparse
import asyncio
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SLIDES = (ROOT / "docs" / "demo" / "trinh_bay.html").as_uri()
OUT_DIR = ROOT / "output" / "videos"
WORK = OUT_DIR / "_promo_work"
W, H = 1600, 900


# ---------- thao tác ----------
def slide(n):
    def act(p):
        p.goto(f"{SLIDES}#{n}")
        p.wait_for_timeout(1200)
    return act


def next_slide(p):
    p.keyboard.press("ArrowRight")
    p.wait_for_timeout(700)


def s5(p):
    p.wait_for_timeout(2600)
    p.click('[data-toggle="obj"] button[data-v="0"]')
    p.wait_for_timeout(3000)
    p.click('[data-toggle="obj"] button[data-v="1"]')


def s6(p):
    for v in ("weather_rain_heavy", "weather_fog_dense", "weather_storm"):
        p.wait_for_timeout(2600)
        p.click(f'[data-toggle="weather"] button[data-v="{v}"]')


def s8(p):
    box = p.locator("#cmp").bounding_box()
    y = box["y"] + box["height"] / 2
    p.mouse.move(box["x"] + box["width"] * 0.5, y)
    p.mouse.down()
    p.mouse.move(box["x"] + box["width"] * 0.08, y, steps=40)
    p.wait_for_timeout(400)
    p.mouse.move(box["x"] + box["width"] * 0.92, y, steps=70)
    p.mouse.up()


def wait_idle(p):
    p.wait_for_function("document.querySelector('#loading').classList.contains('hidden')", timeout=120000)


def run_gen(p):
    p.click("#btn-run")
    p.wait_for_timeout(300)
    wait_idle(p)


def scroll_to(p, sel):
    p.eval_on_selector(sel, "e => e.scrollIntoView({behavior: 'smooth', block: 'center'})")
    p.wait_for_timeout(900)


def web_open(url):
    def act(p):
        p.goto(url)
        p.wait_for_function("document.querySelectorAll('#sel-scenario option').length > 0", timeout=120000)
        p.select_option("#sel-scenario", "moto_swarm")
        p.wait_for_timeout(600)
    return act


def w13(p):
    p.wait_for_timeout(1500)
    run_gen(p)


def w14(p):
    p.click('#seg-color button[data-v="object"]')


def w15(p):
    p.click('#seg-disp button[data-v="original"]')
    p.wait_for_timeout(1800)
    p.click('#seg-disp button[data-v="both"]')
    p.wait_for_timeout(1800)
    p.click('#seg-disp button[data-v="augmented"]')
    p.click('#seg-view button[data-v="driver"]')


def w16(p):
    p.click('#seg-view button[data-v="iso"]')
    p.select_option("#sel-weather", "rain")
    p.select_option("#sel-level", "heavy")
    run_gen(p)
    p.wait_for_timeout(1500)
    p.select_option("#sel-weather", "storm")
    run_gen(p)
    scroll_to(p, "#cams-gen")


def w17(p):
    scroll_to(p, "#btn-run")
    p.select_option("#sel-weather", "none")
    p.click('#seg-mode button[data-v="custom"]')
    p.click('#custom-presets button[data-p="Đường quê"]')
    run_gen(p)
    scroll_to(p, "#cams-gen")
    p.wait_for_timeout(1500)
    p.uncheck("#chk-paste")
    wait_idle(p)
    p.wait_for_timeout(1500)
    p.check("#chk-paste")
    wait_idle(p)


def w18(p):
    scroll_to(p, "#btn-run")
    run_gen(p)
    scroll_to(p, "#cams-gen")


def w19(p):
    scroll_to(p, "#dl")
    p.eval_on_selector("#dl", "e => { e.style.outline = '3px solid #22d3ee'; e.style.borderRadius = '12px'; }")


def segments(url):
    return [
        (slide(1), None, "Đây là một khung hình LiDAR thật, chụp trên đường phố. Những điểm màu hồng là hai mươi chiếc xe máy "
                   "mà hệ thống của chúng tôi vừa đưa vào. Chào mừng đến với N3-08C: dữ liệu giao thông Việt Nam, "
                   "sinh ra từ dữ liệu thật."),
        (next_slide, None, "Xe tự lái hôm nay học từ những bộ dữ liệu quay ở nước ngoài. Trong nuScenes, bộ dữ liệu nổi "
                       "tiếng nhất, không có một khung hình nào có bầy từ năm xe máy trở lên. Kết quả là detector chỉ "
                       "tìm ra năm mươi lăm phần trăm xe máy, trong khi ô tô là chín mươi ba phần trăm."),
        (next_slide, None, "Chúng tôi không đi thu thập thêm. Chúng tôi sinh ra dữ liệu mới từ dữ liệu đã có, qua năm bước: "
                       "cắt vật thể thật, đặt đúng lòng đường, dán theo vật lý của cảm biến, phủ thời tiết, và tự kiểm tra."),
        (next_slide, None, "Mỗi điểm được chèn vào đều tuân theo bốn luật vật lý của LiDAR: một tia chỉ cho một điểm, mật độ "
                       "giảm theo khoảng cách, chỉ thấy mặt quay về cảm biến, và luôn chạm mặt đất."),
        (next_slide, s5, "Khi ẩn các xe được chèn, bạn sẽ thấy những vùng trống phía sau mỗi xe. Đó là vùng bị che, "
                         "đúng như một cảm biến thật nhìn thấy."),
        (next_slide, s6, "Cùng một cảnh, bốn kiểu trời. Tia laser càng đi xa càng dễ mất, theo định luật Beer Lambert. "
                         "Giọt mưa và hạt sương gần xe tạo ra nhiễu trên từng tia."),
        (next_slide, None, "Xe ba gác, xe máy chở ba người, xe hàng rong, bò và chó trên đường. Những tình huống rất Việt Nam "
                       "mà chưa bộ dữ liệu nào có."),
        (next_slide, s8, "Hình ảnh camera cũng được đồng bộ, khớp với từng điểm LiDAR."),
        (next_slide, None, "Kết quả: một trăm phần trăm cảnh sinh ra hợp lệ, không chồng vật thể, đúng mặt đường. Chín nghìn "
                       "bốn trăm cảnh mỗi giờ, chỉ trên một chiếc laptop."),
        (next_slide, None, "So với cách dán thẳng thông thường, mật độ điểm khớp với vật thể thật và mọi vật thể đều chạm mặt đất."),
        (next_slide, None, "Quan trọng nhất: dữ liệu này tìm ra điểm mù của AI. Trong bầy xe máy, khả năng phát hiện giảm mười "
                       "hai điểm. Thêm giông bão, giảm tiếp hai mươi ba điểm. Xe máy chở hàng cồng kềnh chỉ được nhận "
                       "đúng hai mươi chín phần trăm."),
        (next_slide, None, "Tất cả nằm trong một ứng dụng web. Mời bạn xem trực tiếp."),
        (web_open(url), w13, "Chọn một khung hình thật, chọn kịch bản bầy xe máy, và bấm Sinh. Chưa tới một giây, gần hai mươi "
                        "xe máy được chèn, mỗi xe đều có sẵn nhãn 3D."),
        (None, w14, "Các điểm tím là xe được chèn. Hệ thống đã tự loại hàng trăm vị trí sai, chỉ giữ lại những vị trí hợp lệ."),
        (None, w15, "Trước và sau khi chèn. Đây là góc nhìn từ chính chiếc xe tự lái."),
        (None, w16, "Thêm mưa to, rồi giông bão. Point cloud mất điểm ở xa, và cả sáu camera cũng phủ mưa tương ứng."),
        (None, w17, "Kịch bản đường quê: bò, xe bò, chó. Bật đồng bộ ảnh, và vật thể xuất hiện trên camera đúng vị trí, "
              "đúng góc nhìn."),
        (None, w18, "Mỗi lần bấm là một cảnh hoàn toàn mới."),
        (None, w19, "Dữ liệu tải về ngay ở định dạng nuScenes, PCD và PLY, kèm nhãn, sẵn sàng để huấn luyện."),
        (slide(13), None, "N3-08C: dữ liệu cho xe tự lái hiểu đường phố Việt Nam. Cảm ơn bạn đã theo dõi."),
    ]


CAPTION_JS = """t => {
  let d = document.getElementById('__cap');
  if (!d) { d = document.createElement('div'); d.id = '__cap'; document.body.appendChild(d);
    Object.assign(d.style, {position: 'fixed', left: '50%', bottom: '28px', transform: 'translateX(-50%)', zIndex: 99999,
      maxWidth: '1320px', padding: '10px 22px', borderRadius: '12px', background: 'rgba(5,8,15,.78)', color: '#fff',
      font: '600 24px/1.4 "Segoe UI", system-ui, sans-serif', textAlign: 'center', pointerEvents: 'none'}); }
  d.textContent = t;
  d.style.display = t ? 'block' : 'none';
  if (!window.__beat) {  // trang đứng yên thì Chrome không gửi khung hình mới -> phụ đề đổi mà video không ghi lại
    const b = document.createElement('div'); document.body.appendChild(b);
    Object.assign(b.style, {position: 'fixed', right: 0, top: 0, width: '1px', height: '1px', zIndex: 99999});
    let k = 0; window.__beat = () => { b.style.background = (k++ % 2) ? '#070b14' : '#070b15'; requestAnimationFrame(window.__beat); };
    requestAnimationFrame(window.__beat);
  } }"""


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def duration(path):
    out = subprocess.run([ffmpeg(), "-i", str(path)], capture_output=True, text=True, errors="ignore").stderr
    h, m, s = re.search(r"Duration: (\d+):(\d+):([\d.]+)", out).groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


async def tts(texts, voice, rate):
    import edge_tts
    paths = []
    for i, t in enumerate(texts):
        f = WORK / f"a{i:02d}.mp3"
        if not f.is_file():
            await edge_tts.Communicate(t, voice, rate=rate).save(str(f))
        paths.append(f)
    return paths


def record(segs, audio, url):
    from playwright.sync_api import sync_playwright
    durs = [duration(a) for a in audio]
    starts = []
    with sync_playwright() as pw:
        try:
            br = pw.chromium.launch(channel="chrome", args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        except Exception:
            br = pw.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        ctx = br.new_context(viewport={"width": W, "height": H}, record_video_dir=str(WORK),
                             record_video_size={"width": W, "height": H})
        page = ctx.new_page()
        t0 = time.monotonic()
        for i, ((prep, during, text), d) in enumerate(zip(segs, durs)):
            if prep:
                prep(page)                # chuẩn bị (chuyển slide, mở trang) trước khi đọc
            page.evaluate(CAPTION_JS, text)
            start = time.monotonic() - t0
            starts.append(start)
            if during:
                during(page)              # thao tác trong lúc đọc
                page.evaluate(CAPTION_JS, text)  # trang có thể vừa đổi DOM
            print(f"[{i + 1:02d}/{len(segs)}] {start:6.1f}s  audio {d:4.1f}s  thao tác {time.monotonic() - t0 - start:4.1f}s")
            left = d + 0.5 - (time.monotonic() - t0 - start)
            if left > 0:
                page.wait_for_timeout(int(left * 1000))
        page.evaluate(CAPTION_JS, "")
        page.wait_for_timeout(1500)
        total = time.monotonic() - t0
        video = page.video.path()
        ctx.close()
        br.close()
    return Path(video), starts, total


def mux(video, audio, starts, out):
    cmd = [ffmpeg(), "-y", "-i", str(video)]
    for a in audio:
        cmd += ["-i", str(a)]
    parts = [f"[{i + 1}:a]adelay={int(s * 1000)}|{int(s * 1000)}[a{i}]" for i, s in enumerate(starts)]
    mix = "".join(f"[a{i}]" for i in range(len(audio))) + f"amix=inputs={len(audio)}:normalize=0[aout]"
    cmd += ["-filter_complex", ";".join(parts + [mix]), "-map", "0:v", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30",
            "-c:a", "aac", "-b:a", "160k", str(out)]
    subprocess.run(cmd, check=True, capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="vi-VN-HoaiMyNeural")
    ap.add_argument("--rate", default="+5%")
    ap.add_argument("--url", default="http://localhost:8060")
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    segs = segments(args.url)
    audio = asyncio.run(tts([t for _, _, t in segs], args.voice, args.rate))
    print("TTS xong:", len(audio), "đoạn, tổng", round(sum(duration(a) for a in audio), 1), "s")
    video, starts, total = record(segs, audio, args.url)
    out = OUT_DIR / "demo_gioi_thieu.mp4"
    mux(video, audio, starts, out)
    print(f"saved {out}  ({total:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
