import cv2
import numpy as np

def bgsegmMask(bg,img):
	fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
	fgmask = fgbg.apply(bg)
	# fgmask = fgbg.apply(bg)
	# fgmask = fgbg.apply(bg)
	fgmask = fgbg.apply(img)
	return fgmask

def cropBox(img): # Input: Image with only the object
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	_,_,r = cv2.split(img)
	x,y,w,h = cv2.boundingRect(r)
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
	mask = bgsegmMask(bg,img)
	mask = inflateErode(mask,40)
	mask = erodeInflateSmart(mask,20)
	mask = contourMask(mask)
	img[mask==0] = (0,255,0)
	img = cropBox(img)
	return img

def createComposite(img,mask):
	while 1:
		single_layer = mask.copy()
		single_layer[:,:,:] = 0
		single_layer[:,:,1] = 255
		ymin = np.random.randint(single_layer.shape[0]-img.shape[0]+1)
		ymax = ymin+img.shape[0]
		xmin = np.random.randint(single_layer.shape[1]-img.shape[1]+1)
		xmax = xmin+img.shape[1]
		single_layer[ymin:ymax,xmin:xmax] = img
		# Use the single layer as a mask against the final layer
		# if the result has any non pure green pixels, try again
		mask_test = mask.copy() 
		mask_test[np.where((single_layer==[0,255,0]).all(axis=2))] = [0,255,0]
		if np.mean(mask_test[:,:,1])==255:
			mask[ymin:ymax,xmin:xmax] = img
			break
	width = mask.shape[1]
	height = mask.shape[0]
	return width,height,xmin,ymin,xmax,ymax