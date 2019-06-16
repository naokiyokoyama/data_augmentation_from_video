import sys
import os
import glob
import argparse
from multiprocessing import Process
from utils.generate_composites_from_videos import main as generate_composites

this_dir = os.path.dirname(os.path.abspath(__file__))

def print_progress(img_dir, total_composites, indices_tuples):
	num_pics = 0
	while num_pics != total_composites:
		if os.path.exists(img_dir):
			num_pics = len(glob.glob(img_dir+'/*.JPG'))
			sys.stdout.write('\r'+str(num_pics)+' of '+str(total_composites)+' generated')
			sys.stdout.flush()

def print_progress(img_dir, total_composites, indices_tuples):
	num_pics = 0
	while num_pics != total_composites:
		if os.path.exists(img_dir):
			all_composites = glob.glob(os.path.join(img_dir, '*.JPG'))
			num_pics = len(all_composites)
			progress = ''
			for i in indices_tuples:
				pics_per_generator = [j for j in all_composites if i[0] <= int(os.path.basename(j).split('.')[0]) <= i[1]]
				progress += '%d for %d-%d, ' % (len(pics_per_generator), i[0], i[1])
			sys.stdout.write('%d of %d generated: ' % (num_pics, total_composites) + progress[:-2] + '\r')
			sys.stdout.flush()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('num_composites', type=int,
	                    help='total amount of composites to generate')
	parser.add_argument('num_generators', type=int,
	                    help='amount of generators to create composites')
	parser.add_argument('dataset_name',
	                    help='name for the dataset')
	parser.add_argument('path_to_png_folders',
	                    help='path to directory containing folders of PNGs')
	args = parser.parse_args()
	
	# Parse arguments
	total_composites = args.num_composites
	num_generators   = args.num_generators
	name             = args.dataset_name 
	pngs_dir         = args.path_to_png_folders
	batch_size       = int(float(total_composites)/float(num_generators))

	# Figure out number of classes
	pngs_folders = [i for i in glob.glob(os.path.join(pngs_dir,'*')) if os.path.isdir(i)]
	num_classes = 0 
	for png_folder in pngs_folders:
		png_folder_id = int(os.path.basename(png_folder).split('-')[0])
		num_classes = max(num_classes, png_folder_id+1)

	# Create directories
	ann_dir = os.path.join(this_dir, 'data/generated_pictures/annotations_'+name)
	img_dir = os.path.join(this_dir, 'data/generated_pictures/images_'+name)
	for directory in [ann_dir, img_dir]:
		if not os.path.exists(directory):
			os.makedirs(directory)

	# Spawn composite generator scripts
	processes = list()
	indices_tuples = list()
	for i in xrange(num_generators):
		start_index = i*batch_size
		end_index = min((i+1)*batch_size-1, total_composites)
		p = Process(target=generate_composites, args=(pngs_dir, start_index, end_index, name, False))
		p.start()
		processes.append(p)
		indices_tuples.append((start_index, end_index))

	# Spawn print progress process
	print_p = Process(target=print_progress, args=(img_dir, total_composites, indices_tuples))
	print_p.start()
	for p in processes:
		p.join()
	print_p.join()