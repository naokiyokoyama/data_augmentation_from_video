# Artificial Training Data Generation for Mask R-CNN

Generate several artifical composites with pixelwise annotations for each object, using videos taken of each object rotating atop a turntable.

## Step 1: Recording videos

You will need: 

* A good camera that won't dynamically adjust to changing colors during recording (disable auto exposure). As far as I know, this is not possible on an iPhone camera, and possibly with other smartphones as well.
* A tripod
* An electric turntable
* A stopwatch

Here are the steps to take for each object:

**You can take multiple videos of an object,** in order to capture perspectives from different camera elevations, or different object orientations (i.e. to capture an object's underside).

1. Set up the camera/tripod/turntable so that when the object is placed on top, it is entirely visible in the frame, and its **shadow is casted away from / not visible to the camera**
2. Take the object **off** the turntable. Start recording, and only **after recording starts**, place the object on the turntable. You have 5 seconds to place the object on the turntable with your hand. 
3. Start the stopwatch after placing the object. Stop recording after 1 minute. Currently, the script only processes the first 57 seconds after the video starts. This is just enough time for an entire rotation (at least for my turntable). It's OK if your video is longer than 57 seconds, but it cannot be shorter.

## Step 2: Extracting the object from the video and generating folders of PNG files

1. Using `vids2pngs.py`, generate a folder of PNG files containing just the object extracted from every frame of your video (between the 5 and 57 second marks). The usage is `python vids2pngs.py <path to .MOV file>`
2. Do this for every object.
3. Once you are happy with your results, rename each of these folders in this format: `<class_id>-<class_name>`. If you have multiple videos of a particular class, then rename that folder to `<class_id>-<viewpoint_id>-<class_name>`. Class IDs must start at 0 and end at N-1, where N is the total number classes; in other words, start from 0 and don't skip numbers. 

## Step 3: Generating the composites and their annotations

1. Use `generate_composites_from_videos.py` to start creating the composites. 
2. 
