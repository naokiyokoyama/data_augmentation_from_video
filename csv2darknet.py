import glob, os
import numpy as np

# Create directory for labels
directory = 'labels'
if not os.path.exists(directory):
	os.makedirs(directory)
# Parse through the CSV
csv_file = open('train.csv')
dataline = csv_file.readline() # Discard header
last_filepath = ''
while 1:
	# Read data
	dataline = csv_file.readline()
	if len(dataline)==0:
		break
	data = dataline.split(',')
	file_path = data[0]
	filename = file_path.split('/')[-1]
	if file_path != last_filepath:
		if last_filepath!='':
			txt_file.close()
		txt_file = open(directory+'/'+filename.split('.')[0]+'.txt','w')
	print data
	class_id = data[3].split('-')[-1]
	width = float(data[1])
	xmin = float(data[4])
	xmax = float(data[6])
	height = float(data[2])
	ymin = float(data[5])
	ymax = float(data[7])
	# Darknet conversions
	xcent = np.mean([xmin,xmax])/width
	ycent = np.mean([ymin,ymax])/height
	width_percent = (xmax-xmin)/width
	height_percent = (ymax-ymin)/height
	new_data = [class_id,xcent,ycent,width_percent,height_percent]
	new_dataline = " ".join(str(x) for x in new_data)
	txt_file.write(new_dataline+'\n')
	last_filepath = file_path
txt_file.close()
csv_file.close()

names_file = open('river2.names','w')
# How many different classes?
num_class = 0
for vid_path in glob.glob('data/videos/*.MOV'):
	if len(vid_path.split('-')) == 2:
		num_class += 1
for x in range(num_class):
	for vid_path in glob.glob('data/videos/'+str(x)+'-*'):
		if len(vid_path.split('-')) == 2:
			names_file.write(vid_path.split('-')[-1].split('.')[0]+'\n')