import cv2
import numpy as np
import time

def cropBox(img): # Input: Image with only the object
	mask = img[:,:,3]
	x,y,w,h = cv2.boundingRect(mask)
	return img[y:y+h,x:x+w]

def inflateErode(mask,size=50):
	mask2 = mask.copy()
	mask2 = cv2.blur(mask2,(size,size))
	mask2[mask2>0] = 255
	mask2 = cv2.blur(mask2,(size,size))
	mask2[mask2<255] = 0
	return mask2

def erodeInflate(mask,size=20):
	mask2 = mask.copy()
	mask2 = cv2.blur(mask2,(size,size))
	mask2[mask2<255] = 0
	mask2 = cv2.blur(mask2,(size,size))
	mask2[mask2>0] = 255
	return mask2

def erodeInflateSmart(mask,size1=20,size2=20):
	mask2 = mask.copy()
	mask2 = cv2.blur(mask2,(size1,size1))
	mask2[mask2<255] = 0
	mask2 = cv2.blur(mask2,(size2,size2))
	mask3 = mask.copy()
	mask3[mask2==0] = 0
	return mask3

def contourMask(mask):
	_,cnt,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	z = np.zeros(mask.shape)
	max_area = 0;
	max_index = 0;
	i=0
	for c in cnt:
		area = cv2.contourArea(c)
		if area > max_area:
			max_area = area
			max_index = i
		i=i+1
	cv2.drawContours(z,cnt,max_index,255,cv2.FILLED)
	return z

def extractObject(bg,img):
	img = cv2.resize(img,(960,540))
	fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
	mask = fgbg.apply(bg)
	mask = fgbg.apply(bg)
	mask = fgbg.apply(img)
	mask = inflateErode(mask,40)
	mask = erodeInflateSmart(mask,20)
	mask = contourMask(mask)
	img = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA)
	img[mask==0] = (0,0,0,0)
	img = cropBox(img)
	return img

def rotateObject(img):
	height = img.shape[0]
	width = img.shape[1]
	hypotenuse = int(np.ceil(np.sqrt(height**2+width**2)))
	blank_image = np.zeros((hypotenuse,hypotenuse,4), np.uint8)
	x_offset = (hypotenuse-width)/2
	y_offset = (hypotenuse-height)/2
	blank_image[y_offset:y_offset+height,x_offset:x_offset+width]=img
	rotation_degrees = np.random.randint(90)*4
	M = cv2.getRotationMatrix2D((hypotenuse/2,hypotenuse/2),rotation_degrees,1)
	dst = cv2.warpAffine(blank_image,M,(hypotenuse,hypotenuse))
	dst = cropBox(dst)
	return dst

def createComposite(img,mask,rcnn_mask,class_name=None,classes_list=None):
	timeup = time.time()+10
	while 1:
		single_layer = np.zeros(mask.shape, np.uint8)
		ymin = np.random.randint(single_layer.shape[0]-img.shape[0]+1)
		xmin = np.random.randint(single_layer.shape[1]-img.shape[1]+1)
		ymax = ymin+img.shape[0]
		xmax = xmin+img.shape[1]
		
		for i in xrange(ymin,ymax):
			for j in xrange(xmin,xmax):
				if img[i-ymin,j-xmin,3] != 0:
					single_layer[i,j] = img[i-ymin,j-xmin]

		''' single_layer is now a black image with the entire object placed
			within in it at a random location. It's then used as a mask 
			against the segmentation mask we have made so far. If any of the objects 
			in the segmentation mask are occluded too much, try again.'''
		mask_test = cv2.bitwise_and(rcnn_mask,rcnn_mask,mask=single_layer[:,:,3])
		
		# Survey the damage done to the so-far-accumulated mask!

		flat_alpha = list(np.ndarray.flatten(rcnn_mask[:,:,3]))
		num_of_objects = max(flat_alpha)

		if num_of_objects > 0:
			original_pixel_counts = [flat_alpha.count(j+1) for j in xrange(num_of_objects)]
			flat_alpha_test = list(np.ndarray.flatten(mask_test[:,:,3]))
			test_pixel_counts = [flat_alpha_test.count(j+1) for j in xrange(num_of_objects)]
			occlusion_percentages = [float(test_pixel_counts[i])/float(original_pixel_counts[i]) for i in xrange(num_of_objects)]
			try_again = max(occlusion_percentages) > 0.40
		else:
			try_again = False

		if not try_again:
			for i in xrange(ymin,ymax):
				for j in xrange(xmin,xmax):
					if img[i-ymin,j-xmin,3] != 0:
						mask[i,j] = img[i-ymin,j-xmin]
			if rcnn_mask is not None:
				mask_pixel_value = max(np.ndarray.flatten(rcnn_mask))+1
				rcnn_mask[np.where((single_layer!=[0,0,0,0]).all(axis=2))] = [0,0,0,mask_pixel_value]
				classes_list.append(class_name)
			break

		if time.time()>timeup:
			ret = False
			return ret,0,0,0,0,0,0

	width = mask.shape[1]
	height = mask.shape[0]
	ret = True
	return ret,width,height,xmin,ymin,xmax,ymax

def generate_rcnn_masks(image_path,rcnn_mask,classes_list):
	_,_,_,alpha = cv2.split(rcnn_mask)
	mask_pixel_value = max(np.ndarray.flatten(alpha))
	for x in xrange(mask_pixel_value):
		mask = alpha.copy()
		mask[mask==mask_pixel_value-x] = 255
		mask[mask!=255] = 0
		class_name = classes_list.pop()
		if max(np.ndarray.flatten(mask)) != 255:
			continue
		instance_number = classes_list.count(class_name)
		mask_path = [image_path.split('.')[0],class_name,str(instance_number)+'.PNG']
		mask_path = '-'.join(mask_path)
		mask_path = mask_path.replace('images_','annotations_')
		cv2.imwrite(mask_path,mask)