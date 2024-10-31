import cv2 as cv
import torch
import depth_pro
import numpy as np 
import matplotlib.cm as cm
import time
import sys

use_cuda = torch.cuda.is_available()
use_mps = torch.backends.mps.is_available()
device = torch.device('cuda' if use_cuda else 'mps' if use_mps else 'cpu')
model,transform = depth_pro.create_model_and_transforms(device=device)
model.eval()

def depthOnVideoSingle(videoPath):
    cam = cv.VideoCapture(videoPath)
    done = False
    showDepth = False
    prev = time.time()
    current = prev
    while not done:
        ret,frame = cam.read()
        if(ret):
            if(showDepth):
                image = transform(frame)
                prediction = model.infer(image)
                depthMap = cm.jet(prediction['depth'].cpu().numpy())
            current = time.time()
            fps = 1/(current-prev)
            prev = current
            h,w = depthMap.shape[:2] if showDepth else frame.shape[:2]
            cv.imshow("",depthMap if showDepth else frame)
            cv.setWindowTitle("",f"{'Depth Map' if showDepth else 'Webcam Frame'} (Resolution: {w}x{h}, FPS: {fps:.2f})")
            key = cv.waitKey(1)
            if(key == 27 or key == 113): # It's Done (press ESC or q)
                done = True
            elif(key == 100): # Change to Depth Map/Webcam (press d)
                cv.destroyAllWindows()
                showDepth = not(showDepth)
        else:
            done = True


if __name__ == '__main__':
    testVideoPath = './Cup.mp4'
    depthOnVideoSingle(testVideoPath)