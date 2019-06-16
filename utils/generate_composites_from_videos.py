import masker, distorter

import glob, os
import cv2 
import numpy as np
import random as rand
import sys

HEADER = 'filename,width,height,class,xmin,ymin,xmax,ymax\n'
NUM_CLASSES_IN_EACH_COMPOSITE = 4

# png_dirs = 'data/videos/'
pngs_dir = '/Volumes/TOSHIBA/wrs/completed/'

def random_extract(class_id):
	possible_paths = glob.glob(pngs_dir+str(class_id)+'-*')
	png_dirs = [i for i in possible_paths if '.' not in i]
	png_dir = png_dirs[np.random.randint(len(png_dirs))]
	img_path = rand.choice(glob.glob(png_dir+'/*'))
	img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
	class_name = png_dir.split('-')[-1].split('.')[0]
	img = masker.rotateObject(img)
	return img,class_name

def main(START_INDEX, END_INDEX, NAME, multithread=False):
	# How many different classes?
	num_class = max([int(vid_path.split('/')[-1].split('-')[0]) for vid_path in glob.glob(pngs_dir+'*') if os.path.isdir(vid_path)])+1
	if not multithread:
		print("Detected %s different classes" % num_class)
		# Create RCNN annotations
		directory = 'data/generated_pictures/annotations_'+NAME
		if not os.path.exists(directory):
			os.makedirs(directory)
		# Create composites
		directory = 'data/generated_pictures/images_'+NAME
		if not os.path.exists(directory):
			os.makedirs(directory)
		
	# For each composite,
	for x in range(START_INDEX,END_INDEX+1):
		# Randomly select background
		bg_path = rand.choice(glob.glob('data/bg/*.JPG'))
		bg = cv2.imread(bg_path)
		bg = cv2.resize(bg,(750,500))
		
		# Create empty background
		layers = np.zeros((bg.shape[0],bg.shape[1],4), np.uint8)
		rcnn_mask = layers.copy()
		
		# For each class,
		chosen_classes = rand.sample(range(num_class),NUM_CLASSES_IN_EACH_COMPOSITE)
		classes_list = list()

		filename = 'data/generated_pictures/images_'+NAME+'/'+str(x)+'.JPG'
		for class_id in chosen_classes:
			ret = False
			while not ret:
				img,class_name = random_extract(class_id)
				img = distorter.resize_by_dim_and_area(img,bg)
				ret,width,height,xmin,ymin,xmax,ymax = masker.createComposite(img,layers,rcnn_mask,class_name,classes_list)
		composite = bg.copy()
		for i in xrange(layers.shape[0]):
			for j in xrange(layers.shape[1]):
				if layers[i,j,3] > 0:
					composite[i,j,:] = layers[i,j,:3]
		composite = distorter.randomGamma(composite,3.5)
		composite = distorter.randomBlur(composite,2)
		# composite = distorter.randomNoise(composite)
		cv2.imwrite(filename,composite)
		masker.generate_rcnn_masks(filename,rcnn_mask,classes_list)
		if not multithread:
			sys.stdout.write('\r'+str(x-START_INDEX)+' of '+str(END_INDEX-START_INDEX+1)+' generated')
			sys.stdout.flush()

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print "Not enough args given!"
		print "Arg1: Starting image index"
		print "Arg2: Ending image index"
		print "Arg3: Desired name of training set"
		sys.exit()
	START_INDEX = int(sys.argv[1])
	END_INDEX = int(sys.argv[2])
	NAME = sys.argv[3]
	main(START_INDEX,END_INDEX,NAME)
