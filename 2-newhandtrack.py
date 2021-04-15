import cv2
import mediapipe as mp
import time     # to count FPS
import HandTrackingModule as htm


previous_time = 0
current_time = 0
cap = cv2.VideoCapture(0)
detector = htm.HandDetector()

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    landmark_list = detector.find_position(img)

    if len(landmark_list) != 0:
        print(landmark_list[4])

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 135, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)