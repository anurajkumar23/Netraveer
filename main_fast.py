import speech_recognition as sr
import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
import math
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "cell phone", "orange", "broccoli",
               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "telephone",
               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
               "teddy bear", "hair drier", "toothbrush"]

object_dimensions = {
    "bird": 0.10,
    "cat": 0.45,
    "backpack": 0.55,
    "umbrella": 0.50,
    "bottle": 0.20,
    "wine glass": 0.25,
    "cup": 0.15,
    "fork": 0.15,
    "knife": 0.25,
    "spoon": 0.15,
    "banana": 0.20,
    "apple": 0.07,
    "sandwich": 0.20,
    "orange": 0.08,
    "chair": 0.50,
    "laptop": 0.40,
    "mouse": 0.10,
    "remote": 0.20,
    "keyboard": 0.30,
    "phone": 0.15,
    "book": 0.18,
    "toothbrush": 0.16
}

app = FastAPI()

class TargetObjectRequest(BaseModel):
    target_object: str
    real_width: float = 0.15

target_object = ""
real_width = 0.15
previous_notification = None
model = YOLO("yolov8n.pt")

def get_last_word(sentence):
    words = sentence.split()
    return words[-1]

def voice_notification(obj_name, direction, distance):
    engine = pyttsx3.init()
    text = "{} is at {}. It is {:.2f} meters away.".format(obj_name, direction, distance)
    engine.say(text)
    engine.runAndWait()

def draw_clock_overlay(img, center_x, center_y, radius):
    overlay = img.copy()
    for i in range(1, 13):
        angle = math.radians(360 / 12 * i - 90)
        x = int(center_x + radius * math.cos(angle))
        y = int(center_y + radius * math.sin(angle))

        if i % 3 == 0:
            thickness = 3
            length = 20
        else:
            thickness = 1
            length = 10

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(overlay, str(i), (x - 10, y + 10), font, 0.5, (0, 255, 0), thickness)
    return overlay

def listen_for_commands():
    global target_object, real_width
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)

    while True:
        try:
            with mic as source:
                audio = recognizer.listen(source)
            command = recognizer.recognize_google(audio).lower()
            logging.info(f"Recognized command: {command}")

            last_word = get_last_word(command)
            if last_word:
                target_object = last_word

                if target_object in object_dimensions:
                    real_width = float(object_dimensions[target_object])
                else:
                    real_width = 0.15
        except sr.UnknownValueError:
            logging.warning("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            logging.error(f"Could not request results from Google Speech Recognition service; {e}")

@app.post("/set_target")
async def set_target_object(request: TargetObjectRequest):
    global target_object, real_width
    target_object = request.target_object.lower()
    real_width = request.real_width
    logging.info(f"Target object set to: {target_object} with real width: {real_width}")
    return {"message": f"Target object set to: {target_object} with real width: {real_width}"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Camera is not opened")
        await websocket.close()
        return

    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    center_x = int(frame_width // 2)
    center_y = int(frame_height // 2)
    radius = min(center_x, center_y) - 30
    clock_overlay = draw_clock_overlay(np.zeros((int(frame_height), int(frame_width), 3), dtype=np.uint8), center_x, center_y, radius)

    global previous_notification

    try:
        while True:
            success, img = cap.read()
            if not success:
                logging.error("Failed to read from camera")
                break

            img = cv2.addWeighted(img, 1, clock_overlay, 0.3, 0)
            results = model.predict(img, stream=True)

            for r in results:
                boxes = r.boxes

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cls = int(box.cls)

                    if class_names[cls].lower() == target_object:
                        camera_width = x2 - x1
                        distance = (real_width * frame_width) / camera_width

                        obj_center_x = (x1 + x2) // 2
                        obj_center_y = (y1 + y2) // 2

                        camera_middle_x = frame_width // 2
                        camera_middle_y = frame_height // 2

                        vector_x = obj_center_x - camera_middle_x
                        vector_y = obj_center_y - camera_middle_y

                        angle_deg = math.degrees(math.atan2(vector_y, vector_x))
                        if angle_deg < 0:
                            angle_deg += 360

                        if 0 <= angle_deg < 30:
                            direction = "3 o'clock"
                        elif 30 <= angle_deg < 60:
                            direction = "4 o'clock"
                        elif 60 <= angle_deg < 90:
                            direction = "5 o'clock"
                        elif 90 <= angle_deg < 120:
                            direction = "6 o'clock"
                        elif 120 <= angle_deg < 150:
                            direction = "7 o'clock"
                        elif 150 <= angle_deg < 180:
                            direction = "8 o'clock"
                        elif 180 <= angle_deg < 210:
                            direction = "9 o'clock"
                        elif 210 <= angle_deg < 240:
                            direction = "10 o'clock"
                        elif 240 <= angle_deg < 270:
                            direction = "11 o'clock"
                        elif 270 <= angle_deg < 300:
                            direction = "12 o'clock"
                        elif 300 <= angle_deg < 330:
                            direction = "1 o'clock"
                        elif 330 <= angle_deg < 360:
                            direction = "2 o'clock"
                        else:
                            direction = "Unknown Clock Position"

                        cv2.putText(img, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(img, "Distance: {:.2f} meters".format(distance), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                        if previous_notification != (direction, distance):
                            notification_thread = threading.Thread(target=voice_notification, args=(target_object, direction, distance))
                            notification_thread.start()
                            previous_notification = (direction, distance)

            _, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            await websocket.send_bytes(frame)

    except WebSocketDisconnect:
        logging.info("WebSocket disconnected")
    finally:
        cap.release()
        await websocket.close()

@app.get("/")
async def get():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>YOLOv8 Object Detection</title>
    </head>
    <body>
        <h1>YOLOv8 Object Detection</h1>
        <img id="video-frame" src="" alt="Video Stream" style="max-width: 100%;">
        <script>
            const ws = new WebSocket("ws://localhost:8000/ws");
            ws.onmessage = function(event) {
                const image = document.getElementById('video-frame');
                const blob = new Blob([event.data], { type: 'image/jpeg' });
                image.src = URL.createObjectURL(blob);
            };
        </script>
    </body>
    </html>
    """)

if __name__ == "__main__":
    listen_thread = threading.Thread(target=listen_for_commands, daemon=True)
    listen_thread.start()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
