import masker, distorter

import glob, os
import cv2 
import numpy as np
import random as rand
import sys

def extract_from_video(video_path):	
	dirname = video_path.split('.')[0]
	if not os.path.exists(dirname):
		os.mkdir(dirname)

	vid = cv2.VideoCapture(video_path)
	_,bg = vid.read()
	bg = cv2.resize(bg,(960,540))
	for x in xrange(52*30): 
		frame = 5*30+x
		vid.set(1,frame)
		_,img = vid.read()
		img = masker.extractObject(bg,img)
		filename = os.path.join(dirname,str(x)+'.PNG')
		cv2.imwrite(filename,img)
	vid.release()

video_path = sys.argv[1]
if len(video_path.split('.')) < 2:
	for vid in glob.glob(video_path+"/*.MOV"):
		extract_from_video(vid)
else:
	extract_from_video(video_path)