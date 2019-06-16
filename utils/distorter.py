import cv2
import numpy as np
import math

def randomGamma(image, maxDistort):
	gamma = 1 + (maxDistort-1)*np.random.random()
	rand = np.random.random()
	if rand > 0.5:
		invGamma = gamma
	else: 
		invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(image, table)

def randomNoise(image):
	row,col,ch= image.shape
	mean = 0
	var = 200
	sigma = var**0.5
	gauss = np.random.normal(mean,sigma,(row,col,ch))
	gauss = gauss.reshape(row,col,ch)
	noisy = image + gauss
	return noisy

def randomBlur(img, maxBlur):
	blur_kernel = 1+int(np.random.random()*maxBlur)
	img = cv2.blur(img,(blur_kernel,blur_kernel))
	return img

def resize_by_dim_and_area(object_img, background_img, upper_dim_bound=0.50, lower_dim_bound=0.05, upper_area_bound=0.20, lower_area_bound=0.05):
	# 1. First, constrain using larger dimension
	# 2. If it's still too large, constrain by area
	object_height,object_width = object_img.shape[:2]
	background_height,background_width = background_img.shape[:2]
	dim_constrain_percentage = lower_dim_bound+np.random.random()*(upper_dim_bound-lower_dim_bound)
	# Constrain using the larger dimension
	if object_height > object_width:
		constrained_height = int(background_height*dim_constrain_percentage)
		constrained_width  = int(object_width*constrained_height/object_height)
	else:
		constrained_width = int(background_width*dim_constrain_percentage)
		constrained_height  = int(object_height*constrained_width/object_height)
	# Constrain by area if its still too large
	background_area = background_height*background_width
	constrained_object_area = constrained_height*constrained_width
	dim_constrain_percentage = lower_dim_bound+np.random.random()*(upper_dim_bound-lower_dim_bound)
	if constrained_object_area > upper_area_bound*background_area:
		area_constrain_percentage = lower_area_bound+np.random.random()*(upper_area_bound-lower_area_bound)
		scale_factor = math.sqrt(area_constrain_percentage*background_area/constrained_object_area)
		constrained_height = int(scale_factor*constrained_height)
		constrained_width = int(scale_factor*constrained_width)

	resized = cv2.resize(object_img,
						 (constrained_width,constrained_height),
						 interpolation=cv2.INTER_AREA)
	# cv2.imshow('',resized);cv2.waitKey();cv2.destroyAllWindows()

	b,g,r,a = cv2.split(resized)
	a[a>0] = 255
	resized = cv2.merge((b,g,r,a))
	return resized

def resizeRandom(img, lower, upper, shp):
	span = upper-lower
	randSize = lower+np.random.random()*span
	bg_h = shp[0]
	new_h = bg_h*randSize
	new_w = img.shape[1]*new_h/img.shape[0];
	resized = cv2.resize(img,(int(new_w),int(new_h)),interpolation=cv2.INTER_AREA)
	resized[resized[:,:,3]<255]=[0,0,0,0]
	return resized