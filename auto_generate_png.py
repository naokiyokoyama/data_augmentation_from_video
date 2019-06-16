import glob
import sys
import os

from utils import vids2pngs

this_dir = os.path.dirname(os.path.abspath(__file__))
videos_dir = os.path.join(this_dir,'data/videos')

if __name__ == '__main__':
	while True:
		videos_files = glob.glob(os.path.join(videos_dir, '*.MOV'))
		for video_file in videos_files:
			no_extension, _ = os.path.splitext(video_file)
			
			# If a directory with the file name of the video does not exist in the same folder,
			if not os.path.isdir(no_extension):
				os.path.mkdir(no_extension)
				basename = os.path.basename(video_file)
				print 'Found unprocessed video: ', basename
				print 'Generating pngs...'
				
				# Make the output folder
				vids2pngs.extract_from_video(
					video_path=video_file, 
					output_dirname=no_extension, 
					start_time=5, 
					end_time=52, 
					frame_increment=3)

				# It's important to break and update video_files because there may be new folders/files after processing
				break
