# The mediapipe website link: https://google.github.io/mediapipe/solutions/hands



import cv2
import mediapipe as mp
import time     # to count FPS

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

previous_time = 0
current_time = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)         # to make sure there is a detection of hand

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            for id, land_mark in enumerate(hand_landmark.landmark):
                # print(id, land_mark)
                height, width, channel = img.shape
                center_x, center_y = int(land_mark.x * width), int(land_mark.y * height)
                print(id, center_x, center_y)
                if id ==0:
                    cv2.circle(img, (center_x, center_y), 10, (255,0,0), cv2.FILLED)


            mpDraw.draw_landmarks(img, hand_landmark, mpHands.HAND_CONNECTIONS)


    current_time = time.time()
    fps = 1/(current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 135, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)