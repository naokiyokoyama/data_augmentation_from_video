import masker, distorter

import glob, os
import cv2 
import numpy as np

HEADER = 'filename,width,height,class,xmin,ymin,xmax,ymax\n'

def randomFrame(class_id):
	for vid_dir in glob.glob('data/videos/0*'):
		dir_id = int((vid_dir.split('/')[-1]).split('-')[0])
		if dir_id!=class_id:
			continue
		vid_path = glob.glob(vid_dir+'/*.MOV')[0]
		vid = cv2.VideoCapture(vid_path)
		_,bg = vid.read()
		bg = cv2.resize(bg,(960,540))
		frameIndex = 5*30+np.random.randint(1000)
		vid.set(1,frameIndex)
		_,img = vid.read()
		img = masker.extractObject(bg,img)
		vid.release()
		class_name = vid_dir.split('-')[-1]
	return img,class_name

def write2CSV(csv,params):
	newline = ",".join(str(x) for x in params)
	csv.write(newline+'\n')


train_csv = open('train.csv','w')
train_csv.write(HEADER)
# How many different classes?
num_class = len(glob.glob('data/videos/0*'))
# Array of background images
bg_list = []
for img_path in glob.glob('data/bg/*.JPG'):
	img = cv2.imread(img_path)
	bg_list.append(img)
# Create 1000 composites
directory = 'data/generated_images/composites'
if not os.path.exists(directory):
	os.makedirs(directory)
# for each composite
for x in range(0,1000):
	bg_index = np.random.randint(len(bg_list))
	bg = bg_list[bg_index]
	layers = bg.copy()
	layers[:,:,:] = 0
	layers[:,:,1] = 255
	# 0-3 instances per class
	num_inst_each_class = [np.random.randint(3) for xx in range(num_class)]
	filename = directory+'/'+str(x)+'.JPG'
	# for each class,
	for class_id in range(num_class):
		# for each instance
		for xxx in range(num_inst_each_class[class_id]):
			img,class_name = randomFrame(class_id)
			img = distorter.resizeRandom(img,0.2,0.4)
			width,height,xmin,ymin,xmax,ymax = masker.createComposite(img,layers)
			label = class_name+'-'+str(class_id)
			write2CSV(train_csv,[filename,width,height,label,xmin,ymin,xmax,ymax])
	composite = bg.copy()
	composite[np.where((layers!=[0,255,0]).all(axis=2))] = layers[np.where((layers!=[0,255,0]).all(axis=2))]
	composite = distorter.randomGamma(composite,1.8)
	composite = distorter.randomBlur(composite,2)
	# composite = distorter.randomNoise(composite)
	cv2.imwrite(filename,composite)