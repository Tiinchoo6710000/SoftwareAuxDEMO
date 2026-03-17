import cv2
import numpy as np
import torch
import threading
import time
from datetime import datetime
from ultralytics import YOLO

from fall_detector import FallDetector
from roi import point_inside_roi, bbox_roi_coverage
from face_recognition_module import load_known_faces, recognize
from alert_manager import trigger_alert
from tracker import SimpleTracker


YOLO_WIDTH = 640
YOLO_HEIGHT = 384
YOLO_EVERY_N = 6
FACE_EVERY_N = 20
DISPLAY_SCALE = 0.5

CONF_THRESHOLD = 0.5
ROI_COVERAGE_THRESHOLD = 0.8
STILL_ALERT_TIME = 3.0

# 🔥 NUEVO (persistencia)
ALERT_HOLD_TIME = 2.5


latest_frame = None
latest_detections = []

running = True
frame_count = 0


print("\n==============================")
print(" SISTEMA DE MONITOREO DEMO IA")
print("==============================\n")

print("Seleccione el modo de detección:\n")
print("1 - Persona en el Suelo")
print("2 - Zona no Permitida\n")

mode = input("Ingrese opción: ")

if mode == "1":
    detection_name = "Persona en el Suelo"
else:
    detection_name = "Zona no Permitida"

print("\nModo seleccionado:", detection_name)


RTSP_URL = "http://192.168.0.225:8080/video"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


model = YOLO("models/yolov8n-pose.pt").to(DEVICE)

fall_detector = FallDetector()
tracker = SimpleTracker()

load_known_faces()


ROIS = []

ret, frame = cap.read()

if not ret:
    print("Error leyendo cámara")
    exit()


roi_points = []


def mouse_callback(event, x, y, flags, param):

    global roi_points

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points.append((x, y))


def select_roi():

    global roi_points

    print("Dibuja el ROI con clicks. ENTER para confirmar.")

    small = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)

    cv2.namedWindow("Seleccionar ROI")
    cv2.setMouseCallback("Seleccionar ROI", mouse_callback)

    while True:

        frame_copy = small.copy()

        for p in roi_points:
            cv2.circle(frame_copy, p, 5, (0, 255, 0), -1)

        if len(roi_points) > 1:
            cv2.polylines(frame_copy, [np.array(roi_points)], False, (0, 255, 0), 2)

        cv2.imshow("Seleccionar ROI", frame_copy)

        key = cv2.waitKey(1)

        if key == 13 and len(roi_points) > 2:
            break

        if key == 27:
            roi_points = []
            break

    cv2.destroyWindow("Seleccionar ROI")

    if len(roi_points) > 2:
        scaled = [(int(x / DISPLAY_SCALE), int(y / DISPLAY_SCALE)) for x, y in roi_points]
        ROIS.append(np.array(scaled))


select_roi()


def camera_thread():

    global latest_frame

    while running:

        ret, frame = cap.read()

        if not ret:
            continue

        latest_frame = frame


def ai_thread():

    global latest_frame
    global latest_detections
    global frame_count

    while running:

        if latest_frame is None:
            time.sleep(0.01)
            continue

        frame = latest_frame.copy()

        frame_count += 1

        if frame_count % YOLO_EVERY_N != 0:
            time.sleep(0.005)
            continue

        frame_small = cv2.resize(frame, (YOLO_WIDTH, YOLO_HEIGHT))

        results = model(frame_small, verbose=False)

        detections = []

        for r in results:

            if r.boxes is None:
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            keypoints = None

            if r.keypoints is not None:
                keypoints = r.keypoints.data.cpu().numpy()

            for i, box in enumerate(boxes):

                if confs[i] < CONF_THRESHOLD:
                    continue

                kp = None

                if keypoints is not None:
                    kp = keypoints[i]

                detections.append({
                    "box": box,
                    "keypoints": kp
                })

        latest_detections = detections

        time.sleep(0.01)


t1 = threading.Thread(target=camera_thread)
t2 = threading.Thread(target=ai_thread)

t1.start()
t2.start()


# 🔥 NUEVO
alert_active = False
last_alert_time = 0


while True:

    if latest_frame is None:
        time.sleep(0.01)
        continue

    frame = latest_frame.copy()

    fh, fw = frame.shape[:2]

    scale_x = fw / YOLO_WIDTH
    scale_y = fh / YOLO_HEIGHT

    detections = []

    for det in latest_detections:

        box = det["box"]
        kp = det["keypoints"]

        x1 = int(box[0] * scale_x)
        y1 = int(box[1] * scale_y)
        x2 = int(box[2] * scale_x)
        y2 = int(box[3] * scale_y)

        if kp is not None:

            kp_scaled = []

            for k in kp:

                x = int(k[0] * scale_x)
                y = int(k[1] * scale_y)
                c = k[2]

                kp_scaled.append((x, y, c))

        else:
            kp_scaled = None

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "keypoints": kp_scaled
        })


    tracks = tracker.update([d["bbox"] for d in detections])

    alert = False


    for track_id, box in tracks:

        x1, y1, x2, y2 = box

        kp = None

        for d in detections:
            if d["bbox"] == box:
                kp = d["keypoints"]

        inside_roi = False
        coverage = 0

        for roi in ROIS:

            coverage = bbox_roi_coverage(box, roi)

            if coverage >= ROI_COVERAGE_THRESHOLD:
                inside_roi = True
                break

        if not inside_roi:
            continue


        if mode == "1":

            if coverage >= ROI_COVERAGE_THRESHOLD and fall_detector.is_person_still(track_id, STILL_ALERT_TIME):
                alert = True

            if kp is not None:

                on_ground = fall_detector.is_person_on_ground(box, kp)

                fall_detector.update(track_id, on_ground, box)

                if fall_detector.is_person_down(track_id):
                    alert = True


        if mode == "2":

            if frame_count % FACE_EVERY_N == 0:

                name = recognize(frame, (x1, y1, x2, y2))

                if name == "Unknown":
                    alert = True


        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


    # 🔥 LÓGICA NUEVA (persistencia)
    current_time = time.time()

    if alert:
        last_alert_time = current_time

        if not alert_active:
            trigger_alert(detection_name)
            alert_active = True

    if current_time - last_alert_time > ALERT_HOLD_TIME:
        alert_active = False


    for roi in ROIS:
        cv2.polylines(frame, [roi], True, (255, 0, 0), 2)


    # 🔥 CAMBIO CLAVE
    if alert_active:

        cv2.rectangle(frame, (0, 0), (500, 50), (0, 0, 255), -1)

        cv2.putText(
            frame,
            detection_name + " DETECTADA",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )


    console = np.zeros((frame.shape[0], 300, 3), dtype=np.uint8)

    now = datetime.now()

    cv2.putText(console, "SISTEMA AUXILIAR DEMO", (0, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(console, "Fecha:", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    cv2.putText(console, now.strftime("%Y-%m-%d"), (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    cv2.putText(console, "Hora:", (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    cv2.putText(console, now.strftime("%H:%M:%S"), (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    cv2.putText(console, "Modo activo:", (10, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    cv2.putText(console, detection_name, (10, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

    status = "ALERTA ACTIVADA" if alert_active else "MONITOREANDO"
    color = (0, 0, 255) if alert_active else (0, 255, 0)

    cv2.putText(console, "Estado:", (10, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    cv2.putText(console, status, (10, 340),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


    display = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)

    combined = np.hstack((display, cv2.resize(console, (300, display.shape[0]))))

    cv2.imshow("Monitor IA", combined)


    if cv2.waitKey(1) & 0xFF == 27:
        running = False
        break


cap.release()
cv2.destroyAllWindows()