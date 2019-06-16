import masker

import os
import cv2 
import sys

FPS = 30

def extract_from_video(video_path, output_dirname, start_time=5, end_time=52, frame_increment=1):	
	vid = cv2.VideoCapture(video_path)
	_, bg = vid.read()
	bg = cv2.resize(bg,(960,540))
	
	start_frame = FPS*start_time
	end_frame   = FPS*end_time
	frame = start_frame
	frames_processed = 0
	total_frames_to_process = int((end_frame-start_frame)/frame_increment)
	filename_char_length = len('%d.PNG'%total_frames_to_process)
	while frame <= end_frame:
		vid.set(1, frame)
		_, img = vid.read()
		img = masker.extractObject(bg, img)
		filename = os.path.join(output_dirname, ('%d.PNG'%frames_processed).zfill(filename_char_length))
		cv2.imwrite(filename, img)

		# Print progress
		frames_processed += 1
		percent_done = frames_processed/total_frames_to_process
		progress_bar = '['+('='*int(percent_done*30))+'>'+('-'*(29-int(percent_done*30)))+']'
		sys.stdout.write('\r'+progress_bar+' %d of %d generated'%(frames_processed, total_frames_to_process))
		sys.stdout.flush()

		frame += frame_increment

	vid.release()
