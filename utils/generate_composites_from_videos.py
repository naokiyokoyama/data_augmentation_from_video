import masker, distorter

import glob, os
import cv2 
import numpy as np
import random as rand
import sys
import argparse

NUM_CLASSES_IN_EACH_COMPOSITE = 4
COMPOSITE_WIDTH  = 750
COMPOSITE_HEIGHT = 500

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def random_extract(pngs_dir, class_id):
	candidate_dirs = [i for i in glob.glob(os.path.join(pngs_dir, '*')) if os.path.isdir(i)]
	rand_dir = rand.choice(candidate_dirs)
	rand_img_path = rand.choice(glob.glob(os.path.join(rand_dir, '*.PNG')) + glob.glob(os.path.join(rand_dir, '*.png')))
	img = cv2.imread(rand_img_path, cv2.IMREAD_UNCHANGED)
	class_name = pngs_dir.split('-')[-1].split('.')[0]
	img = masker.rotateObject(img)
	return img, class_name

def main(pngs_dir, start_index, end_index, name, print_info=True):
	# How many different classes?
	pngs_folders = [i for i in glob.glob(os.path.join(pngs_dir,'*')) if os.path.isdir(i)]
	num_classes = 0 
	for png_folder in pngs_folders:
		png_folder_id = int(os.path.basename(png_folder).split('-')[0])
		num_classes = max(num_classes, png_folder_id+1)

	if print_info:
		print "Detected %s different classes" % num_classes
		
	# Create directories if they don't already exist
	ann_dir = os.path.join(root_dir, 'data/generated_pictures/annotations_'+name)
	img_dir = os.path.join(root_dir, 'data/generated_pictures/images_'+name)
	for directory in [ann_dir, img_dir]:
		if not os.path.exists(directory):
			os.makedirs(directory)
		
	# Background videos folder
	vids_dir = os.path.join(root_dir, 'data/bg')

	# For each composite,
	for x in range(start_index, end_index+1):
		
		# Randomly select a frame from a randomly selected video
		vid_path = rand.choice(glob.glob(os.path.join(vids_dir, '*.MOV')))
		vid = cv2.VideoCapture(vid_path)
		num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
		rand_frame = rand.randint(0,num_frames-1)
		vid.set(1, rand_frame)
		_, bg = vid.read()
		bg = cv2.resize(bg, (COMPOSITE_WIDTH, COMPOSITE_HEIGHT))
		
		# Create empty backgrounds
		layers = np.zeros((COMPOSITE_HEIGHT, COMPOSITE_WIDTH, 4), np.uint8)
		rcnn_mask = layers.copy()
				
		# List to be populated with class names that appear in the composite
		classes_list = list()

		# Randomly select classes to appear in the composite
		chosen_classes = rand.sample(range(num_classes), NUM_CLASSES_IN_EACH_COMPOSITE)
		for class_id in chosen_classes:
			ret = False
			while not ret:
				img, class_name = random_extract(pngs_dir, class_id)
				img = distorter.resize_by_dim_and_area(img, bg)
				ret, width, height, xmin, ymin, xmax, ymax = masker.createComposite(img,
																					layers,
																					rcnn_mask,
																					class_name,
																					classes_list)
		
		# Insert objects into background
		composite = bg.copy()
		for i in xrange(layers.shape[0]):
			for j in xrange(layers.shape[1]):
				if layers[i,j,3] > 0:
					composite[i,j,:] = layers[i,j,:3] # Composite is BGR, layers is BGRA
		
		# Add noise to the composite
		composite = distorter.randomGamma(composite,3.5)
		composite = distorter.randomBlur(composite,2)
		# composite = distorter.randomNoise(composite)
		
		# Save composite
		filename = os.path.join(img_dir, '%d.JPG'%x)
		cv2.imwrite(filename, composite)

		# Save masks for objects that appear in the composite 
		masker.generate_rcnn_masks(filename, rcnn_mask, classes_list)
		if print_info:
			sys.stdout.write('\r'+str(x-start_index+1)+' of '+str(end_index-start_index+1)+' generated')
			sys.stdout.flush()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('start_index', type=int,
	                    help='start index')
	parser.add_argument('end_index', type=int,
	                    help='end index')
	parser.add_argument('dataset_name',
	                    help='name for the dataset')
	parser.add_argument('path_to_png_folders',
	                    help='path to directory containing folders of PNGs')
	args = parser.parse_args()

	# Parse arguments
	start_index      = args.start_index 
	end_index        = args.end_index 
	name             = args.dataset_name 
	pngs_dir         = args.path_to_png_folders

	main(pngs_dir, start_index, end_index, name)
