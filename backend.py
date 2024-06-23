import cv2
import numpy as np
import pyttsx3
import math
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import threading
from ultralytics import YOLO

class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "telephone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

object_dimensions = {
    "bird": "0.10",
    "cat": "0.45",
    "backpack": "0.55",
    "umbrella": "0.50",
    "bottle": "0.20",
    "wine glass": "0.25",
    "cup": "0.15",
    "fork": "0.15",
    "knife": "0.25",
    "spoon": "0.15",
    "banana": "0.20",
    "apple": "0.07",
    "sandwich": "0.20",
    "orange": "0.08",
    "chair": "0.50",
    "laptop": "0.40",
    "mouse": "0.10",
    "remote": "0.20",
    "keyboard": "0.30",
    "phone": "0.14",
    "book": "0.18",
    "toothbrush": "0.16"
}

app = FastAPI()

def voice_notification(obj_name, direction, distance):
    engine = pyttsx3.init()
    text = f"{obj_name.capitalize()} is at {direction}. It is {distance:.2f} meters away."
    engine.say(text)
    engine.runAndWait()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        await websocket.close()
        return

    model = YOLO("yolov8n.pt")
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    center_x = int(frame_width // 2)
    center_y = int(frame_height // 2)
    radius = min(center_x, center_y) - 30

    def draw_clock_overlay(img, center_x, center_y, radius):
        overlay = img.copy()
        for i in range(1, 13):
            angle = math.radians(360 / 12 * i - 90)
            x = int(center_x + radius * math.cos(angle))
            y = int(center_y + radius * math.sin(angle))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(overlay, str(i), (x - 10, y + 10), font, 0.5, (0, 255, 0), 2 if i % 3 == 0 else 1)
        return overlay

    clock_overlay = draw_clock_overlay(np.zeros((int(frame_height), int(frame_width), 3), dtype=np.uint8), center_x, center_y, radius)
    previous_notification = None

    try:
        while True:
            success, img = cap.read()
            if not success:
                break

            img = cv2.addWeighted(img, 1, clock_overlay, 0.3, 0)
            results = model.predict(img, stream=True)

            for r in results:
                boxes = r.boxes

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls)

                    detected_object = class_names[cls].lower()
                    camera_width = x2 - x1
                    distance = (float(object_dimensions.get(detected_object, 0.15)) * frame_width) / camera_width

                    obj_center_x = (x1 + x2) // 2
                    obj_center_y = (y1 + y2) // 2

                    vector_x = obj_center_x - center_x
                    vector_y = obj_center_y - center_y

                    angle_deg = (math.degrees(math.atan2(vector_y, vector_x)) + 360) % 360
                    direction = f"{int((angle_deg + 30) % 360 / 30) + 1} o'clock"

                    cv2.putText(img, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(img, f"Distance: {distance:.2f} meters", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    if previous_notification != (detected_object, direction, distance):
                        notification_thread = threading.Thread(target=voice_notification, args=(detected_object, direction, distance))
                        notification_thread.start()
                        previous_notification = (detected_object, direction, distance)

            _, encoded_img = cv2.imencode('.jpg', img)
            await websocket.send_bytes(encoded_img.tobytes())

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
