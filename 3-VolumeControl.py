import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cam_width, cam_height = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)
present_time = 0

detector = htm.HandDetector(detectconf=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volume_range = volume.GetVolumeRange()          # (-96.0, 0.0, 1.5)
# volume.SetMasterVolumeLevel(-0.0, None)        # For 0, the sound level is 100

min_volume = volume_range[0]
max_volume = volume_range[1]
vol = 0
vol_bar = 350
vol_percent = 0

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    landmarks_list = detector.find_position(img, draw=False)

    if len(landmarks_list) != 0:
        # print(landmarks_list[4], landmarks_list[8])     # 4 and 8 are for Thumb and Index finger tips

        x1, y1 = landmarks_list[4][1], landmarks_list[4][2]
        x2, y2 = landmarks_list[8][1], landmarks_list[8][2]
        center_x, center_y = (x1 + x2) // 2, (y1 + y2)//2

        cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.circle(img, (center_x, center_y), 10, (255, 0, 0), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        # print(length)

        # Hand range is from 25 to 190. It can vary with the distance between hand and cam
        # Volume range is from -96 to 0

        vol = np.interp(length, [25, 190], [min_volume, max_volume])
        vol_bar = np.interp(length, [25, 190], [350, 140])
        vol_percent = np.interp(length, [25, 190], [0, 100])

        print(int(length), vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length < 25:
            cv2.circle(img, (center_x, center_y), 10, (0, 0, 255), cv2.FILLED)

    # To show a volume bar
    cv2.rectangle(img, (40, 140), (65, 350), (0, 0, 255), 4)
    cv2.rectangle(img, (40, int(vol_bar)), (65, 350), (0, 0, 255), cv2.FILLED)
    cv2.putText(img, f'VOL: {int(vol_percent)}%', (11, 390), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    current_time = time.time()
    fps = 1/(current_time - present_time)
    present_time = current_time
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

