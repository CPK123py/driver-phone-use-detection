"""
Driver Phone Detection - Real-Time Test
Detects PERSON + PHONE using YOLOv8 object detection
Triggers Arduino alert when phone is near driver

FIXES vs original:
- Class indices resolved dynamically from model.names (not hardcoded 0/1)
- Confidence threshold lowered to 0.35 for undertrained / small models
- Proximity check uses % of frame width instead of fixed 150px
- verbose=True on first frame so you can see what the model actually detects
- Added --debug flag: prints every detection so you can verify the model works
- Arduino failure no longer crashes startup (was already guarded, kept as-is)
"""

import cv2
import numpy as np
from ultralytics import YOLO
import serial
import time
from collections import deque
import os
import torch
import argparse


class DriverDistractionDetector:
    def __init__(
        self,
        model_path="best.pt",
        arduino_port="COM3",
        arduino_baud=9600,
        conf_threshold=0.35,   # FIX: was hardcoded 0.5 — too strict for small models
        debug=False,
    ):
        self.model_path = model_path
        self.arduino_port = arduino_port
        self.arduino_baud = arduino_baud
        self.arduino = None
        self.alert_cooldown = 0
        self.alert_threshold_frames = 3
        self.detection_queue = deque(maxlen=self.alert_threshold_frames)
        self.conf_threshold = conf_threshold
        self.debug = debug
        self._first_frame = True   # used to print verbose output once

        # ── Load YOLO model ───────────────────────────────────────────────────
        print(f"Loading YOLO model from {model_path}...")
        if not os.path.exists(model_path):
            print(f"⚠️  Model not found at {model_path}")
            print("Train model first: python train_small.py")
            raise FileNotFoundError(f"Model {model_path} not found")

        self.model = YOLO(model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        # ── FIX: resolve class indices dynamically ────────────────────────────
        # Original code hardcoded cls==0 → person, cls==1 → phone.
        # If your dataset has a different order this silently detects nothing.
        names = self.model.names  # dict {int: str}
        print(f"✅ Model loaded on {device}")
        print(f"   Class map: {names}")

        self.person_cls = self._find_class(names, ["person"])
        self.phone_cls  = self._find_class(names, ["phone", "cell phone", "mobile", "cellphone"])

        if self.person_cls is None:
            print("⚠️  WARNING: could not find 'person' class in model. Detection will not work.")
        if self.phone_cls is None:
            print("⚠️  WARNING: could not find 'phone' class in model. Detection will not work.")

        print(f"   Using → person_cls={self.person_cls}, phone_cls={self.phone_cls}")

        # ── Initialize Arduino ────────────────────────────────────────────────
        self.init_arduino()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _find_class(names: dict, candidates: list):
        """Return the integer key whose value matches any candidate (case-insensitive)."""
        for idx, name in names.items():
            if name.lower() in [c.lower() for c in candidates]:
                return idx
        return None

    def init_arduino(self):
        try:
            self.arduino = serial.Serial(self.arduino_port, self.arduino_baud, timeout=1)
            time.sleep(2)
            print(f"✅ Arduino connected on {self.arduino_port}")
        except Exception as e:
            print(f"⚠️  Arduino not available: {e}")
            self.arduino = None

    def send_arduino_alert(self):
        if self.arduino is None or self.alert_cooldown > 0:
            return
        try:
            self.arduino.write(b"ALERT:PHONE_DETECTED\n")
            print("📢 ARDUINO ALERT SENT")
            self.alert_cooldown = 30
        except Exception as e:
            print(f"⚠️  Failed to send Arduino alert: {e}")

    # ── Core detection ────────────────────────────────────────────────────────

    def detect_phone_near_person(self, boxes, frame_width: int):
        """
        Return True if a phone box is spatially close to a person box.

        FIX: distance threshold is now proportional to frame width (20 %)
        instead of a hardcoded 150 px that broke at different resolutions /
        camera distances.
        """
        if self.person_cls is None or self.phone_cls is None:
            return False

        person_boxes, phone_boxes = [], []

        for box in boxes:
            cls  = int(box.cls)
            conf = float(box.conf)
            if conf < self.conf_threshold:
                continue
            xyxy = box.xyxy[0].cpu().numpy()
            if cls == self.person_cls:
                person_boxes.append(xyxy)
            elif cls == self.phone_cls:
                phone_boxes.append(xyxy)

        if self.debug:
            print(f"   [debug] persons={len(person_boxes)}  phones={len(phone_boxes)}")

        if not person_boxes or not phone_boxes:
            return False

        # Proportional threshold — 20 % of frame width
        dist_threshold = frame_width * 0.20   # FIX: was fixed 150 px

        for phone_box in phone_boxes:
            px = (phone_box[0] + phone_box[2]) / 2
            py = (phone_box[1] + phone_box[3]) / 2

            for person_box in person_boxes:
                cx = (person_box[0] + person_box[2]) / 2
                cy = (person_box[1] + person_box[3]) / 2

                dist = np.linalg.norm(np.array([px - cx, py - cy]))
                if self.debug:
                    print(f"   [debug] phone↔person dist={dist:.1f}px  threshold={dist_threshold:.1f}px")
                if dist < dist_threshold:
                    return True

        return False

    def process_frame(self, frame):
        frame_h, frame_w = frame.shape[:2]

        # FIX: on the very first frame run with verbose=True so you can see
        #      raw model output in the terminal and confirm it detects anything.
        verbose = self._first_frame
        results = self.model(frame, verbose=verbose)
        self._first_frame = False

        is_distracted  = False
        person_count   = 0
        phone_count    = 0

        for result in results:
            boxes = result.boxes
            if boxes is None:
                self.detection_queue.append(False)
                continue

            # FIX: use resolved class indices, not hardcoded 0/1
            person_count = sum(
                1 for b in boxes
                if int(b.cls) == self.person_cls and float(b.conf) > self.conf_threshold
            )
            phone_count = sum(
                1 for b in boxes
                if int(b.cls) == self.phone_cls and float(b.conf) > self.conf_threshold
            )

            if self.debug and len(boxes):
                for b in boxes:
                    print(
                        f"   [debug] cls={int(b.cls)} "
                        f"({self.model.names[int(b.cls)]})  "
                        f"conf={float(b.conf):.2f}"
                    )

            if self.detect_phone_near_person(boxes, frame_w):
                is_distracted = True
                self.detection_queue.append(True)
            else:
                self.detection_queue.append(False)

        # Trigger alert when queue is full and 2+ frames positive
        if (
            len(self.detection_queue) == self.alert_threshold_frames
            and sum(self.detection_queue) >= 2
        ):
            self.send_arduino_alert()
            is_distracted = True

        annotated_frame = results[0].plot() if results else frame.copy()
        self.alert_cooldown = max(0, self.alert_cooldown - 1)

        return annotated_frame, is_distracted, person_count, phone_count

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run_live_detection(self, camera_id=0):
        print("=" * 60)
        print("DRIVER PHONE DETECTION - LIVE TEST")
        print("=" * 60)
        print(f"\nConf threshold : {self.conf_threshold}")
        print(f"Debug mode     : {self.debug}")
        print("\nPress 'q' to quit | 'r' to reset alerts\n")

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("❌ Failed to open camera!")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        frame_count = 0
        fps_timer   = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            try:
                annotated_frame, is_distracted, person_count, phone_count = self.process_frame(frame)
            except Exception as e:
                print(f"Error processing frame: {e}")
                annotated_frame = frame.copy()
                is_distracted, person_count, phone_count = False, 0, 0

            # ── Overlay ───────────────────────────────────────────────────────
            status       = "PHONE DETECTED!" if is_distracted else "Safe Driving"
            status_color = (0, 0, 255)       if is_distracted else (0, 255, 0)
            cv2.putText(annotated_frame, status,      (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 2)
            cv2.putText(annotated_frame, f"Person: {person_count} | Phone: {phone_count}",
                        (10, 70),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame,
                        f"Queue: {sum(self.detection_queue)}/{len(self.detection_queue)}",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

            if frame_count % 10 == 0:
                elapsed = time.time() - fps_timer
                fps = 10 / elapsed
                fps_timer = time.time()
                print(
                    f"FPS: {fps:.1f} | Person: {person_count} "
                    f"| Phone: {phone_count} | Alert: {is_distracted}"
                )

            cv2.imshow("Driver Phone Detection", annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                self.detection_queue.clear()
                self.alert_cooldown = 0
                print("Alerts reset.")

        cap.release()
        cv2.destroyAllWindows()
        if self.arduino:
            self.arduino.close()
        print(f"\n✅ Tested {frame_count} frames")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug",  action="store_true", help="Print every detection to terminal")
    parser.add_argument("--conf",   type=float, default=0.35, help="Confidence threshold (default 0.35)")
    parser.add_argument("--camera", type=int,   default=0,    help="Camera index (default 0)")
    parser.add_argument("--port",   type=str,   default="COM3", help="Arduino serial port")
    args = parser.parse_args()

    detector = DriverDistractionDetector(
        model_path="best.pt",
        arduino_port=args.port,
        arduino_baud=9600,
        conf_threshold=args.conf,
        debug=args.debug,
    )
    detector.run_live_detection(camera_id=args.camera)


if __name__ == "__main__":
    main()