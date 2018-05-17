import cv2
import numpy as np

def randomGamma(image,maxDistort):
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

def randomBlur(img,maxBlur):
	blur_kernel = 1+int(np.random.random()*maxBlur)
	img = cv2.blur(img,(blur_kernel,blur_kernel))
	return img

def resizeRandom(img,lower,upper,shp):
	span = upper-lower
	randSize = lower+np.random.random()*span
	bg_h = shp[0]
	new_h = bg_h*randSize
	new_w = img.shape[1]*new_h/img.shape[0];
	resized = cv2.resize(img,(int(new_w),int(new_h)),interpolation=cv2.INTER_AREA)
	resized[resized[:,:,3]<255]=[0,0,0,0]
	return resized