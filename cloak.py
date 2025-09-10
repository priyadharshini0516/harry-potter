import cv2
import numpy as np
import time

# ------------------ Camera Setup ------------------
cap = cv2.VideoCapture(0)
time.sleep(2)  # wait for camera to warm up

# Capture background
background = 0
for i in range(60):
    ret, background = cap.read()
background = np.flip(background, axis=1)  # flip for mirror effect

# ------------------ Main Loop ------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = np.flip(frame, axis=1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ------------------ Pink color range ------------------
    lower_pink1 = np.array([145, 50, 50])
    upper_pink1 = np.array([170, 255, 255])
    lower_pink2 = np.array([160, 50, 50])
    upper_pink2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_pink1, upper_pink1)
    mask2 = cv2.inRange(hsv, lower_pink2, upper_pink2)

    mask = mask1 + mask2

    # Remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3,3), np.uint8))

    # Inverse mask
    mask_inv = cv2.bitwise_not(mask)

    # Segment cloak and background
    res1 = cv2.bitwise_and(background, background, mask=mask)
    res2 = cv2.bitwise_and(frame, frame, mask=mask_inv)

    # Combine images
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    # Display
    cv2.imshow('Invisible Pink Cloak', final_output)

    # Exit
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
