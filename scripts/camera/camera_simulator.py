#!/usr/bin/env python3

# Network Camera Simulator Script.
# Generates a running clock as video image/stream.
#
# Requirements: pip install av opencv-python numpy flask
# This also needs a running MediaMTX instance: https://github.com/bluenviron/mediamtx/releases
# See installation manual: https://mediamtx.org/docs/kickoff/install
#
# Start MediaMTX first, then run this script. It will push an RTSP stream to MediaMTX.
#
#  Camera Simulator Exposes these sources:
#  - HTTP Static Image : http://127.0.0.1:5000/image.jpg
#  - HTTP MJPEG Stream : http://127.0.0.1:5000/stream.mjpeg
#  - RTSP Stream       : rtsp://127.0.0.1:8554/live  (via MediaMTX)
#
# In rayforge, open you machine settings, choose the "Camera" tab, and add a new camera source
# for the above stream types you want to test.

import threading
import time
from fractions import Fraction

import av
import cv2
import numpy as np
from flask import Flask, Response

# Configuration
WIDTH = 640
HEIGHT = 480
FPS = 30

RTSP_URL = "rtsp://127.0.0.1:8554/live"

app = Flask(__name__)


def generate_frame():
    """Generate an OpenCV/numpy frame (BGR) with a border and current time."""
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)

    cv2.rectangle(img, (10, 10), (WIDTH - 10, HEIGHT - 10), (255, 255, 255), 2)

    cv2.line(
        img,
        (WIDTH // 2 - 20, HEIGHT // 2),
        (WIDTH // 2 + 20, HEIGHT // 2),
        (0, 0, 255),
        1,
    )
    cv2.line(
        img,
        (WIDTH // 2, HEIGHT // 2 - 20),
        (WIDTH // 2, HEIGHT // 2 + 20),
        (0, 0, 255),
        1,
    )

    t = time.localtime()
    t_str = time.strftime("%H:%M:%S", t)
    ms = int((time.time() % 1) * 1000)
    time_text = f"{t_str}.{ms:03d}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    text_size = cv2.getTextSize(time_text, font, font_scale, thickness)[0]

    text_x = (WIDTH - text_size[0]) // 2
    text_y = (HEIGHT + text_size[1]) // 2

    cv2.putText(
        img,
        time_text,
        (text_x, text_y),
        font,
        font_scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA,
    )

    return img


# --- 1. HTTP Static Image Endpoint ---
@app.route("/image.jpg")
def static_image():
    frame = generate_frame()
    success, encoded_image = cv2.imencode(".jpg", frame)
    if not success:
        return "Error generating image", 500
    return Response(encoded_image.tobytes(), mimetype="image/jpeg")


def mpeg_generator():
    while True:
        frame = generate_frame()
        success, encoded_image = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        )
        if not success:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + encoded_image.tobytes()
            + b"\r\n"
        )
        time.sleep(1.0 / FPS)


# --- 2. HTTP MJPEG Stream Endpoint ---
@app.route("/stream.mjpeg")
def mjpeg_stream():
    return Response(
        mpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# --- 3. RTSP Push to MediaMTX via PyAV (no ffmpeg subprocess) ---
def run_rtsp_publisher():
    """
    Encode frames as H.264 and publish them as an RTSP stream to an already
    running MediaMTX instance. MediaMTX is the RTSP server here; this script
    behaves like an RTSP client/publisher, such as an IP camera.
    """
    while True:
        container = None
        stream = None
        try:
            print(f"[RTSP] Connecting and publishing to {RTSP_URL} ...")
            container = av.open(
                RTSP_URL,
                mode="w",
                format="rtsp",
                options={"rtsp_transport": "tcp"},
            )

            stream = container.add_stream("libx264", rate=FPS)
            stream.width = WIDTH
            stream.height = HEIGHT
            stream.pix_fmt = "yuv420p"
            stream.codec_context.time_base = Fraction(1, FPS)
            stream.options = {
                "preset": "ultrafast",
                "tune": "zerolatency",
            }

            frame_duration = 1.0 / FPS
            pts = 0

            print("[RTSP] Publishing started.")

            while True:
                loop_start = time.time()

                img = generate_frame()  # BGR numpy array
                video_frame = av.VideoFrame.from_ndarray(img, format="bgr24")
                video_frame.pts = pts
                pts += 1

                for packet in stream.encode(video_frame):
                    container.mux(packet)

                elapsed = time.time() - loop_start
                sleep_time = frame_duration - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as e:  # noqa: BLE001
            print(f"[RTSP] Error: {e}. Retrying in 2 seconds...")
        finally:
            if container is not None:
                if stream is not None:
                    try:
                        # Flush any buffered packets
                        for packet in stream.encode(None):
                            container.mux(packet)
                    except Exception as e:  # noqa: BLE001
                        print(f"[RTSP] Error flushing stream: {e}")
                try:
                    container.close()
                except Exception as e:  # noqa: BLE001
                    print(f"[RTSP] Error closing stream: {e}")
            time.sleep(2)


if __name__ == "__main__":
    rtsp_thread = threading.Thread(target=run_rtsp_publisher, daemon=True)
    rtsp_thread.start()

    print("==================================================")
    print(" Camera Simulator Active:")
    print(" - HTTP Static Image : http://127.0.0.1:5000/image.jpg")
    print(" - HTTP MJPEG Stream : http://127.0.0.1:5000/stream.mjpeg")
    print(f" - RTSP Stream       : {RTSP_URL}  (via MediaMTX)")
    print(" Note: start MediaMTX (./mediamtx) BEFORE running this script.")
    print("==================================================")

    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
