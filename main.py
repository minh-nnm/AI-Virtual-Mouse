import cv2
import numpy as np
import time
import HandTrackingModule as ht
import autopy


wCam, hCam = 640, 480
frameR = 100
smoothening = 7

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = ht.handDetector(maxHands= 1)
wScr, hScr = autopy.screen.size()
#print(wScr, hScr)

while True:
    # 1. Find hand landmarks
    success, frame = cap.read()
    if not success:
        break

    frame = detector.findHands(frame)
    lmList, bbox = detector.findPosition(frame)
    #print(lmList, bbox)

    # 2. Lấy đầu ngón trỏ và ngón giữa.
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        #print(x1, y1, x2, y2)

        # 3. Kiểm tra ngón tay nào đang lên
        fingers = detector.fingersUp()
        #print(fingers)

        cv2.rectangle(frame, (frameR, frameR), (wCam - frameR, hCam - frameR),
                      (255, 0, 255), 2)

        # 4. Chỉ Ngón trỏ: Chế độ Di chuyển
        if fingers[1] == 1 and fingers[2] == 0:
            # 5. Chuyển đổi Tọa độ
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # 6. Làm mịn giá trị
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # 7. Di chuyển chuột
            autopy.mouse.move(wScr - x3, y3)
            cv2.circle(frame, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # 8. Cả ngón trỏ và ngón giữa đều lên: Chế độ Nhấp chuột
        if fingers[1] == 1 and fingers[2] == 1:
            # 9. Tìm khoảng cách giữa các ngón tay
            length, frame, LineInfo = detector.findDistance(8, 12, frame)
            print(length)
            # 10. Nhấp chuột nếu khoảng cách ngắn
            if length < 40:
                cv2.circle(frame, (LineInfo[4], LineInfo[5]),
                           15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 255, 0), 3)

    cv2.imshow("Output", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
