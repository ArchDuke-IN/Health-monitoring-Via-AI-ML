import cv2
import numpy as np
from collections import deque

class JaundiceAnemiaDetector:
    def __init__(self):
        self.jaundice_history = deque(maxlen=60)
        self.anemia_history = deque(maxlen=60)

    def get_sclera_roi(self, frame, face_landmarks, w, h):
        LEFT_SCLERA = [33, 7, 163, 144, 145, 153, 154, 155, 133]
        RIGHT_SCLERA = [362, 382, 381, 380, 374, 373, 390, 249, 263]

        rois = []
        for eye_indices in [LEFT_SCLERA, RIGHT_SCLERA]:
            coords = []
            for idx in eye_indices:
                lm = face_landmarks.landmark[idx]
                coords.append((int(lm.x * w), int(lm.y * h)))

            x_coords = [p[0] for p in coords]
            y_coords = [p[1] for p in coords]
            x1 = max(0, min(x_coords) - 2)
            y1 = max(0, min(y_coords) - 2)
            x2 = min(w, max(x_coords) + 2)
            y2 = min(h, max(y_coords) + 2)

            roi = frame[y1:y2, x1:x2]
            if roi.size > 0 and roi.shape[0] > 3 and roi.shape[1] > 3:
                rois.append((roi, (x1, y1, x2, y2)))

        return rois

    def get_lip_roi(self, frame, face_landmarks, w, h):
        upper_lip = face_landmarks.landmark[13]
        lower_lip = face_landmarks.landmark[14]
        mouth_open_dist = abs(upper_lip.y - lower_lip.y) * h

        if mouth_open_dist > 8:
            return None, None, True

        LIP_INDICES = [61, 40, 37, 0, 267, 270, 291,
                       321, 314, 17, 84, 91]
        coords = []
        for idx in LIP_INDICES:
            lm = face_landmarks.landmark[idx]
            coords.append((int(lm.x * w), int(lm.y * h)))

        x_coords = [p[0] for p in coords]
        y_coords = [p[1] for p in coords]
        x1 = max(0, min(x_coords))
        y1 = max(0, min(y_coords))
        x2 = min(w, max(x_coords))
        y2 = min(h, max(y_coords))

        roi = frame[y1:y2, x1:x2]
        return roi, (x1, y1, x2, y2), False

    def filter_sclera_pixels(self, roi):
        if roi is None or roi.size == 0:
            return None, None
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        sclera_mask = (gray > 100) & (gray < 245)
        if np.sum(sclera_mask) < 5:
            return None, None
        return roi, sclera_mask

    def get_white_reference(self, frame):
        h, w = frame.shape[:2]
        corner = frame[10:40, w - 40:w - 10]
        if corner.size == 0:
            return np.array([200.0, 200.0, 200.0])
        mean_val = np.mean(corner, axis=(0, 1))
        mean_val = np.clip(mean_val, 0.0, 255.0)
        return mean_val.astype(np.float64)

    def detect_jaundice(self, sclera_rois, frame, calibrator):
        cb_scores = []

        white_ref = self.get_white_reference(frame)
        white_ref_clipped = np.clip(white_ref, 0, 255).astype(np.uint8)
        white_ycrcb = cv2.cvtColor(
            white_ref_clipped.reshape(1, 1, 3),
            cv2.COLOR_BGR2YCrCb
        )[0, 0]
        white_cb = float(white_ycrcb[2])

        for roi, _ in sclera_rois:
            if roi is None or roi.size == 0:
                continue

            filtered, mask = self.filter_sclera_pixels(roi)
            if filtered is None or mask is None:
                continue

            ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
            y_ch = ycrcb[:, :, 0]
            cb_ch = ycrcb[:, :, 2].astype(np.float64)

            bright_mask = y_ch > 100
            if np.sum(bright_mask) < 5:
                continue

            mean_cb = float(np.mean(cb_ch[bright_mask]))
            corrected_cb = mean_cb - (white_cb - 128.0)
            cb_scores.append(corrected_cb)

        if not cb_scores:
            return "Unknown", (128, 128, 128), 0

        avg_cb = float(np.mean(cb_scores))
        self.jaundice_history.append(avg_cb)
        smoothed = float(np.mean(self.jaundice_history))

        if smoothed > 135:
            return "High Risk", (0, 0, 255), smoothed
        elif smoothed > 125:
            return "Mild", (0, 165, 255), smoothed
        else:
            return "Normal", (0, 255, 0), smoothed

    def detect_anemia(self, lip_roi, calibrator):
        if lip_roi is None or lip_roi.size == 0:
            return "Unknown", (128, 128, 128), 0

        ycrcb = cv2.cvtColor(lip_roi, cv2.COLOR_BGR2YCrCb)
        y_ch = ycrcb[:, :, 0]
        cr_ch = ycrcb[:, :, 1].astype(np.float64)

        bright_mask = y_ch > 60
        if np.sum(bright_mask) < 5:
            return "Unknown", (128, 128, 128), 0

        non_teeth_mask = bright_mask & (y_ch < 200)
        if np.sum(non_teeth_mask) < 5:
            return "Unknown", (128, 128, 128), 0

        mean_lip_cr = float(np.mean(cr_ch[non_teeth_mask]))
        cr_score = mean_lip_cr / 128.0

        self.anemia_history.append(cr_score)
        smoothed = float(np.mean(self.anemia_history))

        if smoothed < 0.90:
            return "High Risk", (0, 0, 255), smoothed
        elif smoothed < 1.05:
            return "Mild", (0, 165, 255), smoothed
        else:
            return "Normal", (0, 255, 0), smoothed

    def process_frame(self, frame, face_landmarks, w, h, calibrator):
        sclera_rois = self.get_sclera_roi(frame, face_landmarks, w, h)

        for _, (x1, y1, x2, y2) in sclera_rois:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)

        lip_result = self.get_lip_roi(frame, face_landmarks, w, h)
        lip_roi, lip_bbox, mouth_open = lip_result

        if lip_bbox is not None:
            lx1, ly1, lx2, ly2 = lip_bbox
            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (255, 0, 255), 1)

        jaundice_label, jaundice_color, _ = self.detect_jaundice(
            sclera_rois, frame, calibrator
        )

        if mouth_open:
            if len(self.anemia_history) > 0:
                s = float(np.mean(self.anemia_history))
                if s < 0.90:
                    anemia_label = "High Risk"
                    anemia_color = (0, 0, 255)
                elif s < 1.05:
                    anemia_label = "Mild"
                    anemia_color = (0, 165, 255)
                else:
                    anemia_label = "Normal"
                    anemia_color = (0, 255, 0)
            else:
                anemia_label = "Measuring..."
                anemia_color = (200, 200, 200)
        else:
            anemia_label, anemia_color, _ = self.detect_anemia(
                lip_roi, calibrator
            )

        cv2.putText(frame, f"Jaundice:  {jaundice_label}", (20, 290),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, jaundice_color, 2)
        cv2.putText(frame, f"Anemia:    {anemia_label}", (20, 330),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, anemia_color, 2)

        return frame, jaundice_label, anemia_label