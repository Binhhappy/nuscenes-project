# Web demo N3-08C (CPU): sinh kịch bản VN + 3D + 6 camera + tải kết quả + số đo đã có.
# Dữ liệu KHÔNG đóng vào image: mount datasets/ (nuScenes v1.0-mini) và output/ (template bank, kết quả đo).
# Detector trực tiếp không chạy trong container này (cần GPU + OpenPCDet); trang hiển thị kết quả detector đã đo.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOST=0.0.0.0 PORT=8050 NO_BROWSER=1 \
    NUSCENES_DATAROOT=/app/datasets/nuscenes N308C_OUTPUT_DIR=/app/output

# libgl1 / libglib2.0: opencv (nuscenes-devkit kéo opencv-python bản có GUI)
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY webapp/requirements.txt webapp/requirements.txt
RUN pip install --no-cache-dir --retries 10 --timeout 120 -r webapp/requirements.txt

COPY n3-08c/ n3-08c/
COPY webapp/ webapp/
COPY docs/ docs/
COPY assets/ assets/
COPY run_app.py .
RUN mkdir -p webapp/temp_outputs output datasets

EXPOSE 8050
CMD ["python", "run_app.py"]
