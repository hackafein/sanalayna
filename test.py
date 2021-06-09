import cv2
import numpy as np

# Let's load a simple image with 3 black squares
image = cv2.imread('tmp.png')
#cv2.waitKey(0)

# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find Canny edges
edged = cv2.Canny(gray, 30, 200)
#cv2.waitKey(0)

# Finding Contours
# Use a copy of the image e.g. edged.copy()
# since findContours alters the image
_,contours, hierarchy = cv2.findContours(edged,
	cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contur=[]
for i in contours:
    area = cv2.contourArea(i)
    if area> 50:
        a=[i[0],i[1]]
        contur.append(a)
    
    print(area)
contur = np.asarray(contur)
cv2.imshow('Canny Edges After Contouring', edged)
cv2.waitKey(0)

print("Number of Contours found = " + str(len(contur)))

# Draw all contours
# -1 signifies drawing all contours
cv2.drawContours(image, contur, -1, (0, 255, 0), 3)

cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
