import cv2
import numpy as np
from collections import deque

class StressFatigueDetector:
    def __init__(self):
        self.ear_history = deque(maxlen=60)
        self.blink_count = 0
        self.ear_threshold = 0.18
        self.eye_closed = False
        self.eyebrow_history = deque(maxlen=60)
        self.frame_count = 0
        self.last_blink_frame = 0

    def get_head_tilt(self, landmarks, w, h):
        left_ear = landmarks.landmark[234]
        right_ear = landmarks.landmark[454]
        dx = (right_ear.x - left_ear.x) * w
        dy = (right_ear.y - left_ear.y) * h
        return np.arctan2(dy, dx)

    def calculate_ear(self, landmarks, eye_indices, w, h):
        points = []
        for idx in eye_indices:
            lm = landmarks.landmark[idx]
            points.append((lm.x * w, lm.y * h))
        v1 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
        v2 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
        h1 = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
        if h1 == 0:
            return 0.3
        return (v1 + v2) / (2.0 * h1)

    def calculate_eyebrow_raise(self, landmarks, w, h):
        face_top = landmarks.landmark[10]
        face_bottom = landmarks.landmark[152]
        face_height = abs(face_bottom.y - face_top.y) * h
        if face_height == 0:
            return 20

        tilt = self.get_head_tilt(landmarks, w, h)
        normal_x = np.sin(tilt)
        normal_y = np.cos(tilt)

        left_brow = landmarks.landmark[70]
        left_eye = landmarks.landmark[159]
        right_brow = landmarks.landmark[300]
        right_eye = landmarks.landmark[386]

        left_dx = (left_eye.x - left_brow.x) * w
        left_dy = (left_eye.y - left_brow.y) * h
        left_dist = abs(left_dx * normal_x + left_dy * normal_y)

        right_dx = (right_eye.x - right_brow.x) * w
        right_dy = (right_eye.y - right_brow.y) * h
        right_dist = abs(right_dx * normal_x + right_dy * normal_y)

        avg_dist = (left_dist + right_dist) / 2
        return (avg_dist / face_height) * 100

    def update_blink(self, ear):
        self.frame_count += 1
        if ear < self.ear_threshold:
            if not self.eye_closed:
                if self.frame_count - self.last_blink_frame > 12:
                    self.eye_closed = True
                    self.blink_count += 1
                    self.last_blink_frame = self.frame_count
        else:
            self.eye_closed = False

    def assess_fatigue(self, ear_ratio, blink_rate):
        if ear_ratio < 0.70:
            return "High", (0, 0, 255)
        elif ear_ratio < 0.82:
            return "Moderate", (0, 165, 255)
        else:
            if blink_rate < 3:
                return "Moderate", (0, 165, 255)
            return "Low", (0, 255, 0)

    def assess_stress(self, eyebrow_ratio, blink_rate):
        if eyebrow_ratio > 1.20 and blink_rate > 22:
            return "High", (0, 0, 255)
        elif eyebrow_ratio > 1.12 or blink_rate > 18:
            return "Moderate", (0, 165, 255)
        else:
            return "Low", (0, 255, 0)

    def process_frame(self, frame, face_landmarks, w, h, calibrator):
        LEFT_EYE = [362, 385, 387, 263, 373, 380]
        RIGHT_EYE = [33, 160, 158, 133, 153, 144]

        left_ear = self.calculate_ear(face_landmarks, LEFT_EYE, w, h)
        right_ear = self.calculate_ear(face_landmarks, RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2.0

        self.ear_history.append(avg_ear)
        self.update_blink(avg_ear)

        eyebrow_raise = self.calculate_eyebrow_raise(face_landmarks, w, h)
        self.eyebrow_history.append(eyebrow_raise)

        seconds_elapsed = max(self.frame_count / 30, 1)
        blink_rate = (self.blink_count / seconds_elapsed) * 60

        avg_ear_smooth = np.mean(self.ear_history)
        avg_eyebrow = np.mean(self.eyebrow_history)

        if calibrator.is_calibrated and calibrator.baseline_ear is not None:
            ear_ratio = avg_ear_smooth / max(calibrator.baseline_ear, 0.01)
            eyebrow_ratio = avg_eyebrow / max(
                calibrator.baseline_eyebrow, 0.01
            )
        else:
            ear_ratio = avg_ear_smooth / 0.28
            eyebrow_ratio = avg_eyebrow / 10.0

        fatigue_label, fatigue_color = self.assess_fatigue(
            ear_ratio, blink_rate
        )
        stress_label, stress_color = self.assess_stress(
            eyebrow_ratio, blink_rate
        )

        cv2.putText(frame, f"Fatigue:  {fatigue_label}", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, fatigue_color, 2)
        cv2.putText(frame, f"Stress:   {stress_label}", (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, stress_color, 2)
        cv2.putText(frame, f"EAR: {avg_ear_smooth:.2f}", (20, 235),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"Blinks/min: {blink_rate:.0f}", (20, 255),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return frame, stress_label, fatigue_label, avg_ear, eyebrow_raise