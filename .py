import cv2
import mediapipe as mp
import numpy as np
from math import hypot

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mpDraw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam not detected")
    exit()

print("✅ Webcam started. Press Q to quit")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        for i, handLms in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[i].classification[0].label
            mpDraw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            thumb = handLms.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index = handLms.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            tx, ty = int(thumb.x * w), int(thumb.y * h)
            ix, iy = int(index.x * w), int(index.y * h)

            cv2.circle(frame, (tx, ty), 8, (255, 0, 0), -1)
            cv2.circle(frame, (ix, iy), 8, (255, 0, 0), -1)
            cv2.line(frame, (tx, ty), (ix, iy), (0, 255, 0), 3)

            dist = hypot(ix - tx, iy - ty)
            level = int(np.interp(dist, [30, 300], [0, 100]))
            bar = int(np.interp(dist, [30, 300], [400, 150]))

            if label == "Right":
                cv2.rectangle(frame, (50, 150), (80, 400), (255, 0, 0), 2)
                cv2.rectangle(frame, (50, bar), (80, 400), (255, 0, 0), -1)
                cv2.putText(frame, f'VOL {level}%',
                            (20, 440),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 0, 0), 2)
            else:
                x1, x2 = w - 80, w - 50
                cv2.rectangle(frame, (x1, 150), (x2, 400), (255, 0, 0), 2)
                cv2.rectangle(frame, (x1, bar), (x2, 400), (255, 0, 0), -1)
                cv2.putText(frame, f'BRI {level}%',
                            (w - 150, 440),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 0, 0), 2)

    cv2.imshow("Hand Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
