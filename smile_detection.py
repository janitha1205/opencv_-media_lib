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
                point1 = face_land_mark.landmark[306]
                x1 = int(point1.x * w)
                y1 = int(point1.y * h)
                cv2.circle(frame, (x1, y1), 2, (0, 0, 255), 1)
             
                point2 = face_land_mark.landmark[61]
                x2 = int(point2.x * w)
                y2 = int(point2.y * h)
                cv2.circle(frame, (x2, y2), 2, (0, 255, 0), 1)
                dis = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
                print(dis)

        cv2.imshow("Result", frame)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
