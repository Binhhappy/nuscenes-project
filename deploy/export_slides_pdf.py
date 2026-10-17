"""Xuất docs/demo/trinh_bay.html thành PDF 16:9 (1920x1080 mỗi trang) để mở trên Google Drive / in.

Mỗi slide được chụp bằng Chromium (Playwright) sau khi 3D / thanh số liệu đã vẽ xong, rồi ghép thành PDF.
Chạy: python deploy/export_slides_pdf.py   ->  docs/demo/trinh_bay.pdf
"""
import io
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "docs" / "demo" / "trinh_bay.html").as_uri()
OUT = ROOT / "docs" / "demo" / "trinh_bay.pdf"
N_SLIDES = 13
WAIT_MS = {1: 4000, 5: 3500, 6: 3500, 10: 1800, 11: 1800, 13: 4000}  # chờ 3D tải / thanh số liệu chạy xong


def main():
    from playwright.sync_api import sync_playwright
    pages = []
    with sync_playwright() as pw:
        br = pw.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        pg = br.new_page(viewport={"width": 1600, "height": 900}, device_scale_factor=1.2)
        pg.goto(SRC)
        # ẩn thanh tiến độ / số trang và chú thích thao tác chuột (không có ý nghĩa trên PDF)
        pg.add_style_tag(content="#bar,#num,.tip{display:none!important}")
        for i in range(1, N_SLIDES + 1):
            pg.evaluate(f"show({i - 1})")  # đổi #hash không tải lại trang -> gọi thẳng hàm chuyển slide
            pg.wait_for_timeout(WAIT_MS.get(i, 900))
            pg.evaluate("""() => {
              document.querySelectorAll('.foot, .sub, .legend span').forEach(e => {
                if (/kéo chuột|Kéo thanh trượt/.test(e.textContent)) e.textContent = e.textContent
                  .replace(/\\s*·\\s*kéo chuột để xoay/, '').replace(/Kéo thanh trượt để so sánh/, 'Trái: chỉ nhãn 3D · Phải: đồng bộ ảnh');
              });
              const a = document.querySelector('a.cta');
              if (a) a.innerHTML = 'Demo: binhhappy-n3-08c-demo.static.hf.space';
            }""")
            png = pg.screenshot(type="png")
            pages.append(Image.open(io.BytesIO(png)).convert("RGB"))
            print(f"slide {i}/{N_SLIDES}")
        br.close()
    pages[0].save(OUT, save_all=True, append_images=pages[1:], resolution=144, quality=92)
    print(f"saved {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {len(pages)} trang)")


if __name__ == "__main__":
    main()
