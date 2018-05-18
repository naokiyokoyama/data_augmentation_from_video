import generate_composites_from_videos as make_composites
import csv2darknet as setup

import sys, subprocess

if len(sys.argv) < 4:
	print "Must provide args."
	print "Arg1: Number of composites to generate"
	print "Arg2: Desired name of model"
	print "Arg3: Path to darknet directory"
	sys.exit()

COMPOSITES = int(sys.argv[1])
NAME = sys.argv[2]
DARKNET_DIR = sys.argv[3]

make_composites.main(COMPOSITES,NAME)
IMAGES_DIR = 'data/generated_pictures/images_'+NAME+'/'
TRAIN_CSV = NAME+'_train.csv'
setup.main(NAME,IMAGES_DIR,TRAIN_CSV,DARKNET_DIR)
subprocess.check_call(['./darknet','detector','train',NAME+'/'+NAME+'.data',NAME+'/'+NAME+'.cfg','darknet19_448.conv.23'], cwd=DARKNET_DIR)