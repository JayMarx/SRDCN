import cv2

img = cv2.imread('./baby_GT.bmp')
#img2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

img_ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)

img_bic = cv2.resize(img, None, fx=1.0/4, fy=1.0/4, interpolation=cv2.INTER_CUBIC)
cv2.imshow('s', img_bic)


#cv2.imshow('d', img_ycbcr[:,:,::-1])
#img22 = img_ycbcr[:,:,::-1]



#img_float = cv2.normalize(img22.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
#print img_float[:,:,1]

#print img22[:,:,1]
cv2.waitKey()
