import cv2
import mediapipe as mp
import math

webcam = cv2.VideoCapture(0)
webcam.set(3, 1280)
webcam.set(4, 720)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
with mp_pose.Pose(
    static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5
) as pose:

    while True:
        control, frame = webcam.read()
        if control == False:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        h, w, ch = frame.shape
        if result.pose_landmarks:
            wrist = result.pose_landmarks.landmark[16]
            x1 = int(wrist.x * w)
            y1 = int(wrist.y * h)
            cv2.circle(frame, (x1, y1), 2, (0, 0, 255), -1)
            elbow = result.pose_landmarks.landmark[14]
            x2 = int(elbow.x * w)
            y2 = int(elbow.y * h)
            cv2.circle(frame, (x2, y2), 2, (0, 0, 255), -1)
            sholder = result.pose_landmarks.landmark[12]
            x3 = int(sholder.x * w)
            y3 = int(sholder.y * h)
            cv2.circle(frame, (x3, y3), 2, (0, 0, 255), -1)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.line(frame, (x2, y2), (x3, y3), (255, 0, 0), 2)
            ang = (
                (math.atan2((x2 - x1), (y2 - y1)) - math.atan2((x3 - x2), (y3 - y2)))
                * 360
            ) / math.pi
            print(ang)

        cv2.imshow("Result", frame)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
