import cv2 as cv
import torch
import depth_pro
import numpy as np 
import matplotlib.cm as cm
import time
import sys

config = depth_pro.DepthProConfig(
    patch_encoder_preset="dinov2l16_384",
    image_encoder_preset="dinov2l16_384",
    checkpoint_uri="../ml-depth-pro/checkpoints/depth_pro.pt",
    decoder_features=256,
    use_fov_head=True,
    fov_encoder_preset="dinov2l16_384",
)
use_cuda = torch.cuda.is_available()
use_mps = torch.backends.mps.is_available()
device = torch.device('cuda' if use_cuda else 'mps' if use_mps else 'cpu')
model,transform = depth_pro.create_model_and_transforms(device=device,config=config)
model.eval()

def depthOnVideoSingle(videoPath,resize=1):
    cam = cv.VideoCapture(videoPath)
    done = False
    showDepth = True
    prev = time.time()
    current = prev
    while not done:
        ret,frame = cam.read()
        if(ret):
            if(resize != 1):
                frame = cv.resize(frame,(0,0),fx=resize,fy=resize)
            if(showDepth):
                image = transform(frame).to(device)
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

def depthOnVideoLoop(videoPath,resize=1):
    cam = cv.VideoCapture(videoPath)
    done = False
    showDepth = False
    prev = time.time()
    current = prev
    frames = []
    while not done:
        ret,frame = cam.read()
        if(ret):
            if(resize != 1):
                frame = cv.resize(frame,(0,0),fx=resize,fy=resize)
            frames.append(frame)
        else:
            done = True
    while True: 
        for frame in frames:
            if(showDepth):
                image = transform(frame).to(device)
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

if __name__ == '__main__':
    testVideoPath = './Cup.mp4'
    depthOnVideoLoop(testVideoPath)