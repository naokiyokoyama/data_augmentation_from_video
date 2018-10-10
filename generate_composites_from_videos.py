import masker, distorter

import glob, os
import cv2 
import numpy as np
import random as rand
import sys

HEADER = 'filename,width,height,class,xmin,ymin,xmax,ymax\n'
NUM_CLASSES_IN_EACH_COMPOSITE = 4

def random_extract(class_id):
	possible_paths = glob.glob('data/videos/'+str(class_id)+'-*')
	png_dirs = [i for i in possible_paths if '.' not in i]
	png_dir = png_dirs[np.random.randint(len(png_dirs))]
	img_path = rand.choice(glob.glob(png_dir+'/*'))
	img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
	class_name = png_dir.split('-')[-1].split('.')[0]
	img = masker.rotateObject(img)
	return img,class_name

def write2CSV(csv,params):
	newline = ",".join(str(x) for x in params)
	csv.write(newline+'\n')

def main(START_INDEX,END_INDEX,NAME,printing=True):
	train_csv = open(NAME+'_train.csv','w')
	train_csv.write(HEADER)
	# How many different classes?
	num_class = max([int(vid_path.split('/')[-1].split('-')[0]) for vid_path in glob.glob('data/videos/*')])+1
	print("Detected %s different classes" % num_class)

	# Array of background images
	bg_list = [cv2.imread(i) for i in glob.glob('data/bg/*.JPG')]
	
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
		bg = rand.choice(bg_list)
		
		# Create empty background
		layers = np.zeros((bg.shape[0],bg.shape[1],4), np.uint8)
		rcnn_mask = layers.copy()
		
		# For each class,
		chosen_classes = rand.sample(range(num_class),NUM_CLASSES_IN_EACH_COMPOSITE)
		classes_list = list()

		filename = directory+'/'+str(x)+'.JPG'
		for class_id in chosen_classes:
			ret = False
			while not ret:
				img,class_name = random_extract(class_id)
				img = distorter.resize_by_dim_and_area(img,bg)
				ret,width,height,xmin,ymin,xmax,ymax = masker.createComposite(img,layers,rcnn_mask,class_name,classes_list)
				# ret,width,height,xmin,ymin,xmax,ymax = masker.createComposite(img,layers) # FOR DARKNET
			# label = class_name+'-'+str(class_id) # FOR DARKNET
			# write2CSV(train_csv,[filename,width,height,label,xmin,ymin,xmax,ymax]) # FOR DARKNET
		for i in xrange(layers.shape[0]):
			for j in xrange(layers.shape[1]):
				if layers[i,j,3] > 0:
					composite[i,j,:] = layers[i,j,:3]
		composite = distorter.randomGamma(composite,3.5)
		composite = distorter.randomBlur(composite,2)
		# composite = distorter.randomNoise(composite)
		cv2.imwrite(filename,composite)
		masker.generate_rcnn_masks(filename,rcnn_mask,classes_list)
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
