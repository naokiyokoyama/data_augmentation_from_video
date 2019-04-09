import cv2
import glob 
import os

WIDTH = 1024
HEIGHT = 683

# This line ensures the script will work as intended no matter where it is called from
this_dir = os.path.dirname(os.path.abspath(__file__))

for file in glob.glob(this_dir+'/*'):
	# If file ends with jpg or JPG,
	if file.upper().endswith('.JPG'):
		# Resize it
		img = cv2.imread(file)
		img = cv2.resize(img, (WIDTH,HEIGHT), interpolation=cv2.INTER_AREA)
		cv2.imwrite(file,img)