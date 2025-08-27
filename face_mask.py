import cv2
import mediapipe as mp
import math

webcam = cv2.VideoCapture(0)
mp_fac = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
with mp_fac.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.52
) as face_mesh:

    while True:
        control, frame = webcam.read()
        if control == False:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        h, w, ch = frame.shape
        if result.multi_face_landmarks:
            for face_land_mark in result.multi_face_landmarks:

                mp_drawing.draw_landmarks(
                    frame,
                    face_land_mark,
                    mp_fac.FACEMESH_TESSELATION,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        thickness=1, circle_radius=1, color=(30, 20, 100)
                    ),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        thickness=1, circle_radius=1, color=(123, 4, 56)
                    ),
                )

        cv2.imshow("Result", frame)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
