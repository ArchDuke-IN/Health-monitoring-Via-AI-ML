import cv2
import numpy as np
import time
from scipy.signal import butter, filtfilt
from collections import deque

BUFFER_SIZE = 150

class HeartRateDetector:
    def __init__(self):
        self.rgb_buffer = deque(maxlen=BUFFER_SIZE)
        self.timestamp_buffer = deque(maxlen=BUFFER_SIZE)
        self.fps = 30
        self.frame_times = deque(maxlen=60)
        self.bpm = 0
        self.bpm_history = deque(maxlen=15)
        self.last_valid_bpm = 0
        self.prev_gray = None

    def update_fps(self):
        now = time.time()
        self.frame_times.append(now)
        if len(self.frame_times) > 1:
            elapsed = self.frame_times[-1] - self.frame_times[0]
            if elapsed > 0:
                self.fps = len(self.frame_times) / elapsed
        self.fps = max(10, min(self.fps, 60))

    def is_too_much_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return False
        diff = cv2.absdiff(gray, self.prev_gray)
        self.prev_gray = gray
        return np.mean(diff) > 10

    def is_valid_roi(self, roi, calibrator):
        if roi is None or roi.size == 0:
            return False
        if roi.shape[0] < 20 or roi.shape[1] < 20:
            return False
        brightness = np.mean(roi)
        if brightness < 50 or brightness > 230:
            return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mean_h = np.mean(hsv[:, :, 0])
        mean_s = np.mean(hsv[:, :, 1])
        if calibrator.is_calibrated and calibrator.skin_hue is not None:
            hue_tolerance = 12
            if not (calibrator.skin_hue - hue_tolerance
                    <= mean_h
                    <= calibrator.skin_hue + hue_tolerance):
                return False
        else:
            if not (0 <= mean_h <= 30):
                return False
        if mean_s < 15:
            return False
        return True

    def get_roi(self, frame, face_landmarks, w, h, calibrator):
        zones = {
            "Forehead":    (10, 151),
            "Left Cheek":  (234, 93),
            "Right Cheek": (454, 323),
            "Chin":        (18, 152)
        }
        for zone_name, (idx1, idx2) in zones.items():
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
            if not self.is_valid_roi(roi, calibrator):
                continue
            return roi, (x1, y1, x2, y2), zone_name
        return None, None, None

    def bandpass_filter(self, signal, fps):
        low = 0.67
        high = 3.0
        nyq = fps / 2
        if nyq <= high:
            return signal
        b, a = butter(3, [low / nyq, high / nyq], btype='band')
        return filtfilt(b, a, signal)

    def pos_algorithm(self):
        if len(self.rgb_buffer) < BUFFER_SIZE:
            return self.last_valid_bpm

        rgb = np.array(self.rgb_buffer)
        timestamps = np.array(self.timestamp_buffer)

        total_time = timestamps[-1] - timestamps[0]
        if total_time <= 0:
            return self.last_valid_bpm
        actual_fps = len(timestamps) / total_time
        actual_fps = max(10, min(actual_fps, 60))

        R = rgb[:, 0]
        G = rgb[:, 1]
        B = rgb[:, 2]

        Rn = R / (np.mean(R) + 1e-6)
        Gn = G / (np.mean(G) + 1e-6)
        Bn = B / (np.mean(B) + 1e-6)

        S1 = Rn - Gn
        S2 = Rn + Gn - 2 * Bn

        std_s1 = np.std(S1) + 1e-6
        std_s2 = np.std(S2) + 1e-6
        alpha = std_s1 / std_s2
        pulse = S1 + alpha * S2

        pulse = pulse - np.linspace(pulse[0], pulse[-1], len(pulse))
        pulse = pulse - np.mean(pulse)

        try:
            filtered = self.bandpass_filter(pulse, actual_fps)
        except Exception:
            return self.last_valid_bpm

        fft_result = np.fft.rfft(filtered)
        frequencies = np.fft.rfftfreq(len(filtered), d=1.0 / actual_fps)

        valid = (frequencies >= 0.67) & (frequencies <= 3.0)
        if not np.any(valid):
            return self.last_valid_bpm

        peak_freq = frequencies[valid][np.argmax(np.abs(fft_result)[valid])]
        bpm = int(peak_freq * 60)

        if bpm < 45 or bpm > 160:
            return self.last_valid_bpm

        if self.last_valid_bpm > 0 and abs(bpm - self.last_valid_bpm) > 15:
            bpm = int(self.last_valid_bpm * 0.8 + bpm * 0.2)

        self.bpm_history.append(bpm)
        smoothed = int(np.mean(self.bpm_history))
        self.last_valid_bpm = smoothed
        return smoothed

    def process_frame(self, frame, face_landmarks, w, h, calibrator):
        self.update_fps()

        if self.is_too_much_motion(frame):
            cv2.putText(frame, "Hold Still...", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            return frame, self.last_valid_bpm

        roi, bbox, zone_name = self.get_roi(
            frame, face_landmarks, w, h, calibrator
        )

        if roi is None:
            cv2.putText(frame, "No valid ROI", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame, self.last_valid_bpm

        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ROI: {zone_name}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        r_mean = np.mean(roi[:, :, 2])
        g_mean = np.mean(roi[:, :, 1])
        b_mean = np.mean(roi[:, :, 0])
        self.rgb_buffer.append([r_mean, g_mean, b_mean])
        self.timestamp_buffer.append(time.time())

        if len(self.rgb_buffer) == BUFFER_SIZE:
            self.bpm = self.pos_algorithm()

        if self.bpm > 0:
            cv2.putText(frame, f"Heart Rate: {self.bpm} BPM", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            remaining = BUFFER_SIZE - len(self.rgb_buffer)
            cv2.putText(frame,
                        f"Measuring... ({remaining} frames left)",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return frame, self.bpm