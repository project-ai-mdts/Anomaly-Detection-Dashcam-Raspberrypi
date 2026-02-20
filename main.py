import os
import cv2
import time
import numpy as np
from datetime import datetime
from threading import Lock
from flask import Flask, Response, render_template_string, send_from_directory, jsonify
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2

# ---------------- CONFIG ----------------
MODEL_PATH = "/home/sxc-mdts/edge-pothole-detection/models/best_float16.tflite"
OUTPUT_DIR = "/home/sxc-mdts/edge-pothole-detection/detections"
INPUT_SIZE = 320 # Matches your working script
CONF_THRESHOLD = 0.5 # Adjusted for better sensitivity

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- GLOBAL STATE ----------------
total_detections = 0
recent_events = []
is_recording = False
video_writer = None
clip_start_time = 0
last_conf = 0
is_running = True
data_lock = Lock()

live_stats = {
    "cam_fps": "Live",
    "infer_fps": 0,
    "best_score": 0,
    "status": "Running"
}

app = Flask(__name__)

# ---------------- LOAD MODEL ----------------
interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=4)
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

# ---------------- POSTPROCESS ----------------
def postprocess_and_draw(frame, output):
    global last_conf
    h, w, _ = frame.shape
    detections = 0
    max_score_this_frame = 0

    # output shape is (5, 2100) based on your script
    for i in range(min(50, output.shape[1])):
        confidence = output[4, i]
        if confidence > CONF_THRESHOLD:
            detections += 1
            if confidence > max_score_this_frame:
                max_score_this_frame = confidence

            cx, cy, bw, bh = output[0:4, i]
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            cv2.rectangle(frame, (max(0, x1), max(0, y1)), (min(w, x2), min(h, y2)), (0, 123, 255), 3)
            cv2.putText(frame, f"Pothole {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
   
    last_conf = max_score_this_frame
    return detections

# ---------------- LIVE PICAMERA2 ENGINE ----------------
def generate_frames():
    global total_detections, is_recording, video_writer, clip_start_time, is_running, live_stats, recent_events

    # Initialize Picamera2
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()

    try:
        while is_running:
            start_time = time.time()
            frame_rgb = picam2.capture_array()
            # Picamera2 outputs RGB, OpenCV needs BGR for encoding/drawing
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Preprocess for AI
            input_img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
            input_img = input_img.astype(np.float32) / 255.0
            input_img = np.expand_dims(input_img, axis=0)

            # Inference
            interpreter.set_tensor(input_index, input_img)
            interpreter.invoke()
            output = interpreter.get_tensor(output_index)[0]

            detected = postprocess_and_draw(frame, output)

            # Update Stats
            live_stats["infer_fps"] = round(1.0 / (time.time() - start_time), 1)
            if last_conf > live_stats["best_score"]:
                live_stats["best_score"] = round(last_conf, 2)

            # Event Handling: Record 3s clip
            if detected > 0:
                with data_lock:
                    total_detections += detected
                    if not is_recording:
                        is_recording = True
                        clip_start_time = time.time()
                        ts = datetime.now().strftime("%H-%M-%S")
                        filename = f"event_{ts}.mp4"
                        path = os.path.join(OUTPUT_DIR, filename)
                       
                        recent_events.insert(0, {"time": datetime.now().strftime("%H:%M:%S"), "score": round(last_conf, 2), "file": filename})
                        recent_events = recent_events[:10]
                        video_writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (640, 480))

            if is_recording:
                if video_writer: video_writer.write(frame)
                if time.time() - clip_start_time > 3.0:
                    is_recording = False
                    video_writer.release()
                    video_writer = None

            # Encode for Flask
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret: continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
           
    finally:
        picam2.stop()

# ---------------- DASHBOARD HTML (Your Preferred Design) ----------------
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Edge AI Pothole Dashboard</title>
    <style>
        body{margin:0;background:#f5f7fb;font-family:Segoe UI}
        .container{max-width:1400px;margin:25px auto;background:#fff;padding:25px;border-radius:16px;box-shadow:0 10px 30px rgba(0,0,0,.08)}
        .header{display:flex;justify-content:space-between;align-items:center}
        .badge{padding:8px 18px;border-radius:20px;background:#e6f4ff;color:#007bff;font-weight:600}
        .btn{background:#007bff;color:#fff;border:none;border-radius:10px;padding:10px 18px;cursor:pointer;margin-left:5px}
        .grid{display:grid;grid-template-columns:2.2fr 1fr;gap:25px;margin-top:20px}
        .card{background:#fff;border-radius:14px;padding:18px;box-shadow:0 4px 12px rgba(0,0,0,.05)}
        .stats p{margin:10px 0; border-bottom:1px solid #eee; padding-bottom:5px}
        .stats b{float:right; color:#007bff}
        .video{width:100%;border-radius:12px;border:2px solid #007bff}
        .events{max-height:600px;overflow-y:auto}
        .event{display:flex;justify-content:space-between;align-items:center;background:#f8fbff;padding:14px;border-radius:12px;border-left:6px solid #007bff;margin-bottom:12px}
        .score{background:#eaf3ff;padding:4px 10px;border-radius:20px;font-size:13px;color:#007bff}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>Edge AI Pothole Dashboard</h1>
        <div>
            <span id="status-badge" class="badge">Running</span>
            <button class="btn" style="background:#dc3545" onclick="stopAI()">Stop</button>
            <button class="btn" onclick="location.reload()">Refresh</button>
        </div>
    </div>
    <p style="color:gray;">Live stream + recent detections (Auto-updating)</p>
    <div class="grid">
        <div>
            <div class="card stats">
                <p>Camera/Video FPS <b id="cam-fps">0.0</b></p>
                <p>Infer FPS <b id="infer-fps">0.0</b></p>
                <p>Best score <b id="best-score">0.0</b></p>
            </div>
            <div class="card" style="margin-top:15px">
                <img src="{{ url_for('video_feed') }}" class="video">
            </div>
        </div>
        <div class="card">
            <h3>Recent events</h3>
            <div id="event-list" class="events">
                </div>
        </div>
    </div>
</div>

<script>
    function updateData() {
        fetch('/data')
            .then(res => res.json())
            .then(data => {
                document.getElementById('cam-fps').innerText = data.cam_fps;
                document.getElementById('infer-fps').innerText = data.infer_fps;
                document.getElementById('best-score').innerText = data.best_score;
                document.getElementById('status-badge').innerText = data.status;
               
                if(data.status === "Stopped") {
                    document.getElementById('status-badge').style.background = "#ffdce0";
                    document.getElementById('status-badge').style.color = "#dc3545";
                }

                let eventHtml = '';
                data.events.forEach(e => {
                    eventHtml += `
                        <div class="event">
                            <div><b>${e.time}</b><br><span class="score">Score ${e.score}</span></div>
                            <button class="btn" onclick="window.open('/clip/${e.file}')">View</button>
                        </div>`;
                });
                document.getElementById('event-list').innerHTML = eventHtml;
            });
    }

    function stopAI() {
        fetch('/stop', {method: 'POST'}).then(() => alert("Stopping Inference..."));
    }

    // Update the UI every 1 second
    setInterval(updateData, 1000);
</script>
</body>
</html>
"""

# ---------------- FLASK ROUTES ----------------
@app.route('/')
def index(): return render_template_string(DASHBOARD_HTML)

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
def data():
    return jsonify({
        "cam_fps": live_stats["cam_fps"],
        "infer_fps": live_stats["infer_fps"],
        "best_score": live_stats["best_score"],
        "status": live_stats["status"],
        "events": recent_events
    })

@app.route('/clip/<filename>')
def get_clip(filename): return send_from_directory(OUTPUT_DIR, filename)

@app.route('/stop', methods=['POST'])
def stop_ai():
    global is_running
    is_running = False
    live_stats["status"] = "Stopped"
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
