import cv2
from mtcnn2 import MTCNN
from draw_points import *
import os
import numpy as np

def crop_face_with_margin(image, box, margin=10):
    x, y, w, h = box
    h_img, w_img = image.shape[:2]
    x1 = max(x - margin, 0)
    y1 = max(y - margin, 0)
    x2 = min(x + w + margin, w_img)
    y2 = min(y + h + margin, h_img)
    return image[y1:y2, x1:x2]

# Create directory to save cropped faces
os.makedirs("cropped_faces", exist_ok=True)

print('Welcome to Face Detection \n\n Enter 1 to add image manually\n Enter 2 to detect face in Webcam feed')
try:
    n = int(input())
except Exception:
    print('Skipping face_detect script in GUI mode')
    exit(0)

if n != 1 and n != 2:
    print('Wrong Choice')
    exit(0)

count = 0
if n == 1:
    print('Enter complete address of the image')
    addr = '/home/ml/Documents/attendance_dl/21.jpg'
    if not os.path.exists(addr):
        print('Invalid Address')
        exit(0)

    print('Enter Resolution of output image (in heightXwidth format)')
    res = input().split('X')
    img = cv2.imread(addr)
    img = cv2.resize(img, (int(res[1]), int(res[0])))
    ckpts = np.zeros((int(res[0]), int(res[1])), dtype='uint8')

elif n == 2:
    video_capture = cv2.VideoCapture('dataset/Mam.mp4')
    ckpts = np.zeros((480, 840), dtype='uint8')  # Default webcam resolution

detector = MTCNN()
alpha = 0.12
beta = 0.04

while True:
    if n == 2:
        ret, frame = video_capture.read()
        if not ret:
            break
    elif n == 1:
        frame = img

    # Rotate and resize
    m = cv2.getRotationMatrix2D((frame.shape[1] / 2, frame.shape[0] / 2 + 250), -90, 1)
    frame = cv2.warpAffine(frame, m, (frame.shape[1], frame.shape[0]))
    frame = cv2.resize(frame, (840, 480))

    detect = detector.detect_faces(frame)

    if detect:
        for i in range(len(detect)):
            boxes = detect[i]['box']
            keypoints = detect[i]['keypoints']

            # Coordinates check
            try:
                points_to_check = [
                    keypoints['nose'], keypoints['left_eye'],
                    keypoints['right_eye'], keypoints['mouth_left'],
                    keypoints['mouth_right']
                ]
                skip = False
                for pt in points_to_check:
                    if ckpts[pt[1]][pt[0]] != 0:
                        skip = True
                        break
                if skip:
                    continue
            except:
                continue

            # Optional visual marker
            draw_lines(frame, boxes, keypoints, alpha, beta, count)

            # Crop, resize and save face
            cropped_face = crop_face_with_margin(frame, boxes, margin=15)
            try:
                aligned_face = cv2.resize(cropped_face, (160, 160))
                save_path = f"cropped_faces/face_{count}.jpg"
                cv2.imwrite(save_path, aligned_face)
                print(f"[INFO] Saved: {save_path}")
                cv2.imshow("Face Crop", aligned_face)
                count += 1
            except Exception as e:
                print(f"[ERROR] Could not crop/save face {count}: {e}")

    # Show full frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
