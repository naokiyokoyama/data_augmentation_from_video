import masker, distorter

import glob, os
import cv2 
import numpy as np
import random as rand
import sys

HEADER = 'filename,width,height,class,xmin,ymin,xmax,ymax\n'

def random_extract(class_id):
	possible_paths = glob.glob('data/videos/'+str(class_id)+'-*')
	png_dirs = [i for i in possible_paths if '.' not in i]
	png_dir = png_dirs[np.random.randint(len(png_dirs))]
	img_path = rand.choice(glob.glob(png_dir+'/*'))
	img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
	# print img
	# cv2.imshow('',img);cv2.waitKey();cv2.destroyAllWindows()
	class_name = png_dir.split('-')[-1].split('.')[0]
	return img,class_name

def write2CSV(csv,params):
	newline = ",".join(str(x) for x in params)
	csv.write(newline+'\n')

def main(COMPOSITES,NAME):
	train_csv = open(NAME+'_train.csv','w')
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
	# Create RCNN annotations
	directory = 'data/generated_pictures/annotations_'+NAME
	if not os.path.exists(directory):
		os.makedirs(directory)
	# Create composites
	directory = 'data/generated_pictures/images_'+NAME
	if not os.path.exists(directory):
		os.makedirs(directory)
		
	# for each composite,
	for x in range(0,COMPOSITES):
		# select background
		bg_index = np.random.randint(len(bg_list))
		bg = bg_list[bg_index]
		# create empty background
		layers = cv2.cvtColor(bg,cv2.COLOR_BGR2BGRA)
		layers[:,:,:] = 0
		chosen_classes_amount = 4
		chosen_classes = rand.sample(range(num_class),chosen_classes_amount)
		# for each class,
		filename = directory+'/'+str(x)+'.JPG'
		classes_list = list()
		rcnn_mask = layers.copy()
		for class_id in chosen_classes:
			ret = False
			while not ret:
				img,class_name = random_extract(class_id)
				img = distorter.resize_by_dim_and_area(img,bg)
				ret,width,height,xmin,ymin,xmax,ymax = masker.createComposite(img,layers,rcnn_mask,class_name,classes_list)
				# ret,width,height,xmin,ymin,xmax,ymax = masker.createComposite(img,layers)
			label = class_name+'-'+str(class_id)
			write2CSV(train_csv,[filename,width,height,label,xmin,ymin,xmax,ymax])
		composite = cv2.cvtColor(bg,cv2.COLOR_BGR2BGRA)
		for i in xrange(layers.shape[0]):
			for j in xrange(layers.shape[1]):
				if layers[i,j,3] > 0:
					composite[i,j,:] = layers[i,j,:]
		# composite[np.where((layers!=[0,0,0,0]).all(axis=2))] = layers[np.where((layers!=[0,0,0,0]).all(axis=2))]
		# composite[:,:,3] = 255 # Empty
		composite = cv2.cvtColor(composite,cv2.COLOR_BGRA2BGR)
		composite = distorter.randomGamma(composite,3.5)
		composite = distorter.randomBlur(composite,2)
		# composite = distorter.randomNoise(composite)
		cv2.imwrite(filename,composite)
		masker.generate_rcnn_masks(filename,rcnn_mask,classes_list)
		sys.stdout.write('\r'+str(x)+' of '+str(COMPOSITES)+' generated')
		sys.stdout.flush()

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print "Not enough args given!"
		print "Arg1: number of images to generate"
		print "Arg2: desired name of training set"
		sys.exit()
	COMPOSITES = int(sys.argv[1])
	NAME = sys.argv[2]
	main(COMPOSITES,NAME)
