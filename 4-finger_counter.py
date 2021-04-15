import cv2
import time
import os
import HandTrackingModule as htm


wCam = 640
hCam = 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folder_path = "signs"
img_list = os.listdir(folder_path)
print(img_list)

overlay_list = []
for img in img_list:
    image = cv2.imread(f'{folder_path}/{img}')
    overlay_list.append(image)
print(len(overlay_list))

previous_time = 0

detector = htm.HandDetector(detectconf=0.6)

fingertip_ids = [4, 8, 12, 16, 20]
thumb_check = [1, 17]
while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    landmark_list = detector.find_position(img, draw=False)
    # print(landmark_list)

    if len(landmark_list) != 0:
        fingers = []

        # for thumb finger
        if landmark_list[thumb_check[0]][1] < landmark_list[thumb_check[1]][1]:
            # print("left hand")
            cv2.putText(img, 'Left Hand', (10, 185), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if landmark_list[fingertip_ids[0]][1] < landmark_list[fingertip_ids[0] - 2][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            # print("Right hand")
            cv2.putText(img, 'Right Hand', (10, 185), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if landmark_list[fingertip_ids[0]][1] < landmark_list[fingertip_ids[0] - 2][1]:  # [1]=Width,X
                fingers.append(0)
            else:
                fingers.append(1)

        # for other fingers

        for id in range(1, 5):
            if landmark_list[fingertip_ids[id]][2] < landmark_list[fingertip_ids[id]-2][2]:  # [2]=Height,Y
                fingers.append(1)
            else:
                fingers.append(0)

        # print(fingers)

        total_fingers = fingers.count(1)
        print(total_fingers)

        # For just signs
        h, w, c = overlay_list[total_fingers - 1].shape
        img[0:h, 20:w+20] = overlay_list[total_fingers - 1]

        """
        # For gestures
        if total_fingers == 0:
            h, w, c = overlay_list[5].shape
            img[0:h, 20:w+20] = overlay_list[5]

        if total_fingers == 1:
            if landmark_list[fingertip_ids[1]][2] < landmark_list[fingertip_ids[1]-2][2]:
                h, w, c = overlay_list[0].shape
                img[0:h, 20:w+20] = overlay_list[0]
            else:
                h, w, c = overlay_list[7].shape
                img[0:h, 20:w + 20] = overlay_list[7]

        if total_fingers == 2:
            h, w, c = overlay_list[1].shape
            img[0:h, 20:w+20] = overlay_list[1]

        if total_fingers == 3:
            if landmark_list[fingertip_ids[4]][2] > landmark_list[fingertip_ids[4]-2][2]:
                h, w, c = overlay_list[2].shape
                img[0:h, 20:w+20] = overlay_list[2]

            elif landmark_list[fingertip_ids[1]][2] < landmark_list[fingertip_ids[1]-2][2]:
                h, w, c = overlay_list[6].shape
                img[0:h, 20:w+20] = overlay_list[6]

            else:
                h, w, c = overlay_list[8].shape
                img[0:h, 20:w + 20] = overlay_list[8]

        if total_fingers == 4:
            h, w, c = overlay_list[3].shape
            img[0:h, 20:w+20] = overlay_list[3]

        if total_fingers == 5:
            h, w, c = overlay_list[4].shape
            img[0:h, 20:w+20] = overlay_list[4]
    """

        cv2.rectangle(img, (20, 225), (170, 425), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, str(total_fingers), (45, 390), cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 0), 25)
        cv2.putText(img, "Finger Count:", (35, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, f'FPS: {int(fps)}', (485, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Image", img)

    cv2.waitKey(1)