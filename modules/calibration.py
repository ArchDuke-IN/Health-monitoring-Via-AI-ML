import cv2
import numpy as np
from collections import deque

class Calibrator:
    def __init__(self, calibration_frames=90):
        """
        calibration_frames: how many frames to collect before locking baseline
        90 frames at 30fps = 3 seconds
        """
        self.calibration_frames = calibration_frames
        self.is_calibrated = False
        self.frame_count = 0

        # Baselines set during calibration
        self.skin_hue = None          # person's own skin hue
        self.skin_saturation = None   # person's own skin saturation
        self.skin_brightness = None   # person's own skin brightness
        self.baseline_ear = None      # person's normal eye openness
        self.baseline_eyebrow = None  # person's normal eyebrow position
        self.baseline_green = None    # person's skin green channel value

        # Buffers during calibration
        self._hue_buf = []
        self._sat_buf = []
        self._val_buf = []
        self._ear_buf = []
        self._eyebrow_buf = []
        self._green_buf = []

    def reset(self):
        """Call this when a new person steps in front"""
        self.__init__(self.calibration_frames)

    def get_cheek_roi(self, frame, face_landmarks, w, h):
        """
        Gets left cheek region — most reliable skin tone reference
        Away from eyes, lips, hairline
        """
        # Left cheek anchor points
        idx1 = 234
        idx2 = 93

        lm1 = face_landmarks.landmark[idx1]
        lm2 = face_landmarks.landmark[idx2]

        x1 = int(min(lm1.x, lm2.x) * w)
        y1 = int(min(lm1.y, lm2.y) * h)
        x2 = int(max(lm1.x, lm2.x) * w)
        y2 = int(max(lm1.y, lm2.y) * h)

        pad = 10
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        roi = frame[y1:y2, x1:x2]
        return roi

    def collect(self, frame, face_landmarks, w, h, ear, eyebrow_dist):
        """
        Called every frame during calibration period
        Collects skin tone, EAR, eyebrow data
        """
        if self.is_calibrated:
            return

        self.frame_count += 1

        # Get cheek ROI for skin tone
        cheek = self.get_cheek_roi(frame, face_landmarks, w, h)

        if cheek is not None and cheek.size > 0:
            hsv = cv2.cvtColor(cheek, cv2.COLOR_BGR2HSV)
            self._hue_buf.append(np.mean(hsv[:, :, 0]))
            self._sat_buf.append(np.mean(hsv[:, :, 1]))
            self._val_buf.append(np.mean(hsv[:, :, 2]))
            self._green_buf.append(np.mean(cheek[:, :, 1]))

        self._ear_buf.append(ear)
        self._eyebrow_buf.append(eyebrow_dist)

        # Lock in baseline after enough frames
        if self.frame_count >= self.calibration_frames:
            self._lock_baseline()

    def _lock_baseline(self):
        """Finalizes all baseline values"""
        if self._hue_buf:
            self.skin_hue = np.median(self._hue_buf)
            self.skin_saturation = np.median(self._sat_buf)
            self.skin_brightness = np.median(self._val_buf)
            self.baseline_green = np.median(self._green_buf)

        if self._ear_buf:
            self.baseline_ear = np.median(self._ear_buf)

        if self._eyebrow_buf:
            self.baseline_eyebrow = np.median(self._eyebrow_buf)

        self.is_calibrated = True
        print(f"✅ Calibrated — Skin hue: {self.skin_hue:.1f}, "
              f"EAR baseline: {self.baseline_ear:.3f}, "
              f"Eyebrow baseline: {self.baseline_eyebrow:.1f}")

    def draw_calibration_overlay(self, frame, w, h):
        """Shows calibration progress bar on screen"""
        progress = min(self.frame_count / self.calibration_frames, 1.0)
        bar_width = int(w * 0.6)
        bar_x = int(w * 0.2)
        bar_y = int(h * 0.85)

        # Background bar
        cv2.rectangle(frame,
                      (bar_x, bar_y),
                      (bar_x + bar_width, bar_y + 30),
                      (50, 50, 50), -1)

        # Progress fill
        cv2.rectangle(frame,
                      (bar_x, bar_y),
                      (bar_x + int(bar_width * progress), bar_y + 30),
                      (0, 255, 128), -1)

        # Text
        cv2.putText(frame,
                    f"Calibrating... {int(progress * 100)}%",
                    (bar_x + 10, bar_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.putText(frame,
                    "Please look at the camera and stay still",
                    (int(w * 0.15), bar_y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame