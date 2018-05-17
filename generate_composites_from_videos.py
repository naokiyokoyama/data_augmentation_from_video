import masker, distorter

import glob, os
import cv2 
import numpy as np
import random as rand
import sys

COMPOSITES = int(sys.argv[1])

HEADER = 'filename,width,height,class,xmin,ymin,xmax,ymax\n'

def randomFrame(class_id):
	possible_vid_paths = glob.glob('data/videos/'+str(class_id)+'-*')
	vid_path = possible_vid_paths[np.random.randint(len(possible_vid_paths))]
	vid = cv2.VideoCapture(vid_path)
	_,bg = vid.read()
	bg = cv2.resize(bg,(960,540))
	frame = 5*30+np.random.randint(1000)
	vid.set(1,frame)
	_,img = vid.read()
	img = masker.extractObject(bg,img)
	vid.release()
	class_name = vid_path.split('-')[-1].split('.')[0]
	return img,class_name

def write2CSV(csv,params):
	newline = ",".join(str(x) for x in params)
	csv.write(newline+'\n')


train_csv = open('train.csv','w')
train_csv.write(HEADER)
# How many different classes?
num_class = 0
for vid_path in glob.glob('data/videos/*.MOV'):
	if len(vid_path.split('-')) == 2:
		num_class += 1
print("Detected %s different classes" % num_class)
# Array of background images
bg_list = []
for img_path in glob.glob('data/bg/*.JPG'):
	img = cv2.imread(img_path)
	bg_list.append(img)
# Create composites
directory = 'data/generated_images/composites'
if not os.path.exists(directory):
	os.makedirs(directory)
# for each composite,
for x in range(0,COMPOSITES):
	# select background
	bg_index = np.random.randint(len(bg_list))
	bg = bg_list[bg_index]
	# create empty background
	layers = cv2.cvtColor(bg.copy(),cv2.COLOR_BGR2BGRA)
	layers[:,:,:] = 0
	# 0-2 instances per class, but at least one
	# num_inst_each_class = 0
	# while np.sum(num_inst_each_class) == 0:
	# 	num_inst_each_class = [np.random.randint(3) for xx in range(num_class)]
	chosen_classes = rand.sample(range(num_class),5)
	# for each class,
	filename = directory+'/'+str(x)+'.JPG'
	for class_id in chosen_classes:
		img,class_name = randomFrame(class_id)
		img = distorter.resizeRandom(img,0.2,0.4,bg.shape)
		width,height,xmin,ymin,xmax,ymax = masker.createComposite(img,layers)
		composite = cv2.cvtColor(bg,cv2.COLOR_BGR2BGRA)
		label = class_name+'-'+str(class_id)
		write2CSV(train_csv,[filename,width,height,label,xmin,ymin,xmax,ymax])
	# for class_id in range(num_class):
	# 	# for each instance
	# 	for xxx in range(num_inst_each_class[class_id]):
	# 		img,class_name = randomFrame(class_id)
	# 		img = distorter.resizeRandom(img,0.2,0.6,bg.shape)
	# 		width,height,xmin,ymin,xmax,ymax = masker.createComposite(img,layers)
	# 		composite = cv2.cvtColor(bg,cv2.COLOR_BGR2BGRA)
	# 		label = class_name+'-'+str(class_id)
	# 		write2CSV(train_csv,[filename,width,height,label,xmin,ymin,xmax,ymax])
	composite = cv2.cvtColor(bg.copy(),cv2.COLOR_BGR2BGRA)
	composite[np.where((layers!=[0,0,0,0]).all(axis=2))] = layers[np.where((layers!=[0,0,0,0]).all(axis=2))]
	composite[:,:,3] = 255 # Empty
	composite = cv2.cvtColor(composite,cv2.COLOR_BGRA2BGR)
	composite = distorter.randomGamma(composite,1.8)
	composite = distorter.randomBlur(composite,2)
	# composite = distorter.randomNoise(composite)
	cv2.imwrite(filename,composite)