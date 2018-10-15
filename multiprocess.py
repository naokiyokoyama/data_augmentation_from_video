import sys
import os
import numpy as np
import glob
import cv2
from multiprocessing import Process
from generate_composites_from_videos import main as generate_composites

pngs_dir = '/Volumes/TOSHIBA/wrs/completed/'
# pngs_dir = 'data/videos/'

def print_info(total_images, name):
	directory = 'data/generated_pictures/images_'+name
	num_pics = 0
	while num_pics != total_images:
		if os.path.exists(directory):
			num_pics = len(glob.glob(directory+'/*.JPG'))
			sys.stdout.write('\r'+str(num_pics)+' of '+str(total_images)+' generated')
			sys.stdout.flush()

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print "Not enough args given!"
		print "Arg1: Total images"
		print "Arg2: Number of generators"
		print "Arg3: Desired name of training set"
		sys.exit()
	total_images = int(sys.argv[1])
	generators = int(sys.argv[2])
	name = sys.argv[3]
	batch_size = int(np.ceil(float(total_images)/float(generators)))

	num_class = max([int(vid_path.split('/')[-1].split('-')[0]) for vid_path in glob.glob(pngs_dir+'*') if os.path.isdir(vid_path)])+1

	# Create RCNN annotations
	directory = 'data/generated_pictures/annotations_'+name
	if not os.path.exists(directory):
		os.makedirs(directory)
	# Create composites
	directory = 'data/generated_pictures/images_'+name
	if not os.path.exists(directory):
		os.makedirs(directory)

	processes = list()
	for i in xrange(generators):
		start_index = i*batch_size
		end_index = min((i+1)*batch_size-1,total_images)
		p = Process(target=generate_composites, args=(start_index,end_index,name,True))
		p.start()
		processes.append(p)
	print_p = Process(target=print_info, args=(total_images, name))
	print_p.start()
	for p in processes:
		p.join()
	print_p.join()