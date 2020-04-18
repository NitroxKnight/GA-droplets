import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import cv2
import os
from PIL import Image

## grid is like this:
# +--> y
# |
# v
# x
# such that D[px,py] can be written and a image can be viewed from it easily

nx,ny = 720,1280 #resolution of simulation
nx_out,ny_out = 1440,2560 #resolution of video produced

#functions for convertion to a simulation frame to frame of the video
resize = lambda x: np.array(Image.fromarray(x).resize((ny_out,nx_out)))
to_img = lambda x: resize((x.copy()[:,:,None]*255*np.ones((1,1,3))).astype(np.uint8))

#init simulation
D = np.ones((nx,ny)) #dampness
N = 2000 #number of particales
px = np.random.randint(low=0,high=nx,size=(N)) #x positions 
py = np.random.randint(low=0,high=ny,size=(N)) #y positions

#init video
video_name = 'droplets-video.mp4'
height, width = nx_out, ny_out
video = cv2.VideoWriter(video_name, 0, 60, (width,height))
video.write(to_img(D))

#main loop
try:
    for i in tqdm(range(2000)):
        alpha = (10-1)/9
        f = 10
        
        py += 1
        Ddown = D[(px+1)%nx,(py+1)%ny]**f
        Dmiddle = D[px,(py+1)%ny]**f
        Dup = D[(px-1)%nx,(py+1)%ny]**f

        Dsum = np.sum([Ddown,Dmiddle,Dup],axis=0)

        Ddown /= Dsum
        Dmiddle /= Dsum
        Dup /= Dsum
        p = np.array([Ddown,Dmiddle,Dup]) #probablility 

        u = np.random.uniform(size=N)
        down = (u<Ddown).astype(int)
        up = (u>(1-Dup)).astype(int)

        px += down-up
        py %= ny
        px %= nx
        D += (1-D)/150
        D[px,py] /= 10
        D[(px+1)%nx,py] /= 10-alpha*1
        D[(px-1)%nx,py] /= 10-alpha*1
        D[(px+2)%nx,py] /= 10-alpha*2**2
        D[(px-2)%nx,py] /= 10-alpha*2**2
        D[px,(py+1)%ny] /= 10-alpha*1
        D[(px+1)%nx,(py+1)%ny] /= 10-alpha*2
        D[(px-1)%nx,(py+1)%ny] /= 10-alpha*2
        if (i-1)%2==0:
            video.write(to_img(D))
finally: #always release video
    cv2.destroyAllWindows()
    video.release()