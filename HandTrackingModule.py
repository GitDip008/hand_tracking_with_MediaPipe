import cv2
import mediapipe as mp
import time     # to count FPS

class HandDetector():
    def __init__(self, mode = False, maxHands = 2, detectconf = 0.5, trackconf = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectconf = detectconf
        self.trackconf = trackconf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectconf, self.trackconf)
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True,):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)         # to make sure there is a detection of hand

        if self.results.multi_hand_landmarks:
            for hand_landmark in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hand_landmark, self.mpHands.HAND_CONNECTIONS)

        return img

    def find_position(self, img, handNo=0, draw=True):
        landmark_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[handNo]

            for id, land_mark in enumerate(my_hand.landmark):
                # print(id, land_mark)
                height, width, channel = img.shape
                center_x, center_y = int(land_mark.x * width), int(land_mark.y * height)
                # print(id, center_x, center_y)
                landmark_list.append([id, center_x, center_y])
                if draw:
                    cv2.circle(img, (center_x, center_y), 5, (255, 0, 0), cv2.FILLED)

        return landmark_list



def main():
    previous_time = 0
    current_time = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

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




if __name__ == "__main__":
    main()