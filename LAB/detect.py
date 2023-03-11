import numpy as np
from future import print_function
import cv2 as cv

video_path = '/home/massinissa_abboud/Downloads/Pexels Videos 1899131.mp4'

capture = cv.VideoCapture(cv.samples.findFileOrKeep(video_path))
if not capture.isOpened():
    print('Unable to open: ' + video_path)
    exit(0)

while True:
    ret, frame = capture.read()
    cvted = cv.cvtColor(frame, cv.COLOR_BGR2Lab)
    l_star = np.mean(cvted[:, :, 0])
    a_star = np.mean(cvted[:, :, 1])
    b_star = np.mean(cvted[:, :, 2])

    r1 = cvted[:, :, 0] >= l_star
    r2 = cvted[:, :, 1] >= a_star
    r3 = cvted[:, :, 2] >= b_star
    r4 = cvted[:, :, 2] >= cvted[:, :, 1]

    rf = np.array(r1 * r2 * r3 * r4)
    fire_mask = np.ones([frame.shape[0], frame.shape[1], 3])
    frame_with_mask = frame.copy()

    for i in range(3):
        frame_with_mask[:, :, i] = rf
        fire_mask[:, :, i] = rf

    if frame is None:
        break

    THRESOLD = 200

    min_x = 5000
    min_y = 5000
    max_x = -1
    max_y = -1

    ones = np.argwhere(frame_with_mask >= THRESOLD)
    for p in ones:
        y, x = p[0], p[1]
        if x < min_x: min_x = x
        if x > max_x: max_x = x
        if y < min_y: min_y = y
        if y > max_y: max_y = y

    cv.rectangle(frame, pt1=(min_x - 20, min_y - 20), pt2=(max_x + 20, max_y + 20), color=(0, 0, 255), thickness=5)
    cv.rectangle(fire_mask, pt1=(min_x, min_y), pt2=(max_x, max_y), color=(0, 0, 255), thickness=5)
    cv.imshow('gray', frame)
    cv.imshow('fire mask', fire_mask)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break