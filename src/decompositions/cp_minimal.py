import cv2
import numpy as np

video = cv2.VideoCapture("Vid.mov")


val = video.get(cv2.CAP_PROP_FRAME_COUNT)

print(val)

frames = []
while True:
    check, img = video.read()
    if not check:
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    frames.append(gray)

video.release()

#This stacks on the depth axis
video_tensor = np.stack(frames, axis=2)   # shape (H, W, T)

value = video_tensor[:,:,0]

print(value)
print("Tensor shape:", video_tensor.shape)