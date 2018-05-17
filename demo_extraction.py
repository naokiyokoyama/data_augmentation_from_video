import masker

import cv2
import sys

vid = cv2.VideoCapture(sys.argv[1])
_,bg = vid.read()
bg = cv2.resize(bg,(960,540))
frame = 5*30
vid.set(1,frame)
while(1):
	# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
	_,frame = vid.read()
	# fgmask = fgbg.apply(bg)
	# fgmask = fgbg.apply(bg)
	# fgmask = fgbg.apply(bg)
	# fgmask = fgbg.apply(bg)
	# fgmask = fgbg.apply(bg)
	# fgmask = fgbg.apply(bg)
	# fgmask = fgbg.apply(frame)
	fgmask = masker.extractObject(bg,frame)
	# fgmask = masker.erodeInflateSmart(fgmask,size1=10,size2=10)
	# fgmask = masker.inflateErode(fgmask,size=15)
	cv2.imshow('frame',fgmask)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
	    break

vid.release()
cv2.destroyAllWindows()