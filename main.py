import cv2
import mediapipe as mp
import platform
from modules.calibration import Calibrator
from modules.heart_rate import HeartRateDetector
from modules.stress_fatigue import StressFatigueDetector
from modules.jaundice_anemia import JaundiceAnemiaDetector
from modules.face_guide import FaceGuide

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

if platform.system() == "Linux":
    CAMERA_SOURCE = 0
else:
    CAMERA_SOURCE = "http://192.0.0.4:8080/video"

def run():
    cap = cv2.VideoCapture(CAMERA_SOURCE)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Camera opened. Press Q to quit. Press R to reset.")

    calibrator = Calibrator(calibration_frames=90)
    heart_detector = HeartRateDetector()
    stress_detector = StressFatigueDetector()
    jaundice_detector = JaundiceAnemiaDetector()
    face_guide = FaceGuide()

    aligned_frames = 0
    ALIGNMENT_NEEDED = 30

    # Fixed display size
    DISPLAY_W = 640
    DISPLAY_H = 480

    cv2.namedWindow("HealthScan", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("HealthScan", DISPLAY_W, DISPLAY_H)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Force resize every frame to 640x480
            frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
            h, w = frame.shape[:2]

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:

                    frame, is_aligned = face_guide.draw_guide(
                        frame, face_landmarks, w, h
                    )

                    if not calibrator.is_calibrated:

                        if not is_aligned:
                            aligned_frames = 0
                            overlay = frame.copy()
                            cv2.rectangle(overlay, (0, 0), (w, h),
                                          (0, 0, 0), -1)
                            cv2.addWeighted(overlay, 0.3,
                                            frame, 0.7, 0, frame)
                            cv2.putText(frame,
                                        "Align your face with the oval",
                                        (int(w * 0.1), 35),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7, (255, 255, 255), 2)
                        else:
                            aligned_frames += 1
                            cv2.putText(frame,
                                        "Good! Hold still...",
                                        (int(w * 0.28), 35),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7, (0, 255, 0), 2)

                            if aligned_frames >= ALIGNMENT_NEEDED:
                                LEFT_EYE = [362, 385, 387, 263, 373, 380]
                                RIGHT_EYE = [33, 160, 158, 133, 153, 144]
                                left_ear = stress_detector.calculate_ear(
                                    face_landmarks, LEFT_EYE, w, h
                                )
                                right_ear = stress_detector.calculate_ear(
                                    face_landmarks, RIGHT_EYE, w, h
                                )
                                avg_ear = (left_ear + right_ear) / 2.0
                                eyebrow = stress_detector.calculate_eyebrow_raise(
                                    face_landmarks, w, h
                                )
                                calibrator.collect(
                                    frame, face_landmarks,
                                    w, h, avg_ear, eyebrow
                                )

                                progress = min(
                                    calibrator.frame_count /
                                    calibrator.calibration_frames, 1.0
                                )
                                bar_w = int(w * 0.5)
                                bar_x = int(w * 0.25)
                                bar_y = int(h * 0.82)

                                cv2.rectangle(frame,
                                              (bar_x, bar_y),
                                              (bar_x + bar_w, bar_y + 30),
                                              (40, 40, 40), -1)
                                cv2.rectangle(frame,
                                              (bar_x, bar_y),
                                              (bar_x + int(bar_w * progress),
                                               bar_y + 30),
                                              (0, 220, 100), -1)
                                cv2.rectangle(frame,
                                              (bar_x, bar_y),
                                              (bar_x + bar_w, bar_y + 30),
                                              (255, 255, 255), 2)
                                cv2.putText(frame,
                                            f"Calibrating {int(progress * 100)}%",
                                            (bar_x + 10, bar_y + 22),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.6, (0, 0, 0), 2)

                    else:
                        if is_aligned:
                            mp_drawing.draw_landmarks(
                                image=frame,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
                            )

                            frame, bpm = heart_detector.process_frame(
                                frame, face_landmarks, w, h, calibrator
                            )

                            frame, stress, fatigue, ear, eyebrow = (
                                stress_detector.process_frame(
                                    frame, face_landmarks, w, h, calibrator
                                )
                            )

                            frame, jaundice, anemia = (
                                jaundice_detector.process_frame(
                                    frame, face_landmarks, w, h, calibrator
                                )
                            )

                            cv2.putText(frame, "Scanning...", (20, 35),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.8, (0, 255, 0), 2)
                        else:
                            cv2.putText(frame,
                                        "Please re-align your face",
                                        (int(w * 0.1), 35),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7, (0, 165, 255), 2)

            else:
                cx, cy = w // 2, h // 2
                rx, ry = int(w * 0.15), int(h * 0.25)
                cv2.ellipse(frame, (cx, cy), (rx, ry),
                            0, 0, 360, (100, 100, 100), 2)
                cv2.putText(frame,
                            "Position your face in the oval",
                            (int(w * 0.1), cy + ry + 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (200, 200, 200), 2)
                cv2.putText(frame,
                            "No Face Detected",
                            (int(w * 0.28), 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)

                if calibrator.is_calibrated:
                    print("Person left — resetting")
                    calibrator.reset()
                    heart_detector = HeartRateDetector()
                    stress_detector = StressFatigueDetector()
                    jaundice_detector = JaundiceAnemiaDetector()
                    face_guide = FaceGuide()
                    aligned_frames = 0

            cv2.imshow("HealthScan", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("Manual reset")
                calibrator.reset()
                heart_detector = HeartRateDetector()
                stress_detector = StressFatigueDetector()
                jaundice_detector = JaundiceAnemiaDetector()
                face_guide = FaceGuide()
                aligned_frames = 0

    cap.release()
    cv2.destroyAllWindows()

run()