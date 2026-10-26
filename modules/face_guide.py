import cv2
import numpy as np

class FaceGuide:
    def __init__(self):
        self.is_aligned = False
        self.alignment_message = "Position your face in the oval"

    def check_alignment(self, frame, face_landmarks, w, h):
        oval_cx = w // 2
        oval_cy = h // 2
        oval_rx = int(w * 0.15)
        oval_ry = int(h * 0.25)

        nose = face_landmarks.landmark[1]
        nose_x = int(nose.x * w)
        nose_y = int(nose.y * h)

        left_ear = face_landmarks.landmark[234]
        right_ear = face_landmarks.landmark[454]
        face_width = abs(right_ear.x - left_ear.x) * w

        forehead = face_landmarks.landmark[10]
        chin = face_landmarks.landmark[152]
        face_height = abs(chin.y - forehead.y) * h

        dx = nose_x - oval_cx
        dy = nose_y - oval_cy

        too_close = face_width > oval_rx * 2 * 0.95
        too_far = face_width < oval_rx * 2 * 0.50
        off_left = dx < -oval_rx * 0.25
        off_right = dx > oval_rx * 0.25
        off_up = dy < -oval_ry * 0.25
        off_down = dy > oval_ry * 0.25

        if too_close:
            self.alignment_message = "Move Back"
            self.is_aligned = False
        elif too_far:
            self.alignment_message = "Move Closer"
            self.is_aligned = False
        elif off_left:
            self.alignment_message = "Move Right"
            self.is_aligned = False
        elif off_right:
            self.alignment_message = "Move Left"
            self.is_aligned = False
        elif off_up:
            self.alignment_message = "Move Down"
            self.is_aligned = False
        elif off_down:
            self.alignment_message = "Move Up"
            self.is_aligned = False
        else:
            self.alignment_message = "Hold Still"
            self.is_aligned = True

        return self.is_aligned, oval_cx, oval_cy, oval_rx, oval_ry

    def draw_guide(self, frame, face_landmarks, w, h):
        is_aligned, cx, cy, rx, ry = self.check_alignment(
            frame, face_landmarks, w, h
        )

        color = (0, 255, 0) if is_aligned else (0, 0, 255)

        # Draw oval
        cv2.ellipse(frame, (cx, cy), (rx, ry), 0, 0, 360, color, 3)

        # Corner markers
        cv2.line(frame, (cx - 20, cy - ry - 10),
                 (cx + 20, cy - ry - 10), color, 2)
        cv2.line(frame, (cx - 20, cy + ry + 10),
                 (cx + 20, cy + ry + 10), color, 2)

        # Message
        msg = self.alignment_message
        text_size = cv2.getTextSize(
            msg, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        text_x = cx - text_size[0] // 2
        text_y = cy + ry + 45

        cv2.rectangle(frame,
                      (text_x - 10, text_y - 25),
                      (text_x + text_size[0] + 10, text_y + 10),
                      (0, 0, 0), -1)
        cv2.putText(frame, msg, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Crosshair
        cv2.drawMarker(frame, (cx, cy), (255, 255, 255),
                       cv2.MARKER_CROSS, 20, 1)

        return frame, is_aligned