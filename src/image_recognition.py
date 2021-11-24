import cv2
import numpy
import imutils

eft_full = cv2.imread("./images/negative/eft_full.png")
eft_full_grey = cv2.cvtColor(eft_full, cv2.COLOR_BGR2GRAY)
eft_full_smaller = cv2.imread("./images/negative/eft_full.png")
eft_menu = cv2.imread("./images/positive/eft_menu.png")

# for scale in numpy.linspace(0.2, 1.0, 20)[::-1]:


result = cv2.matchTemplate(eft_full, eft_menu, cv2.TM_CCOEFF_NORMED)
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
print(maxVal)
print(maxLoc)

#cv2.imshow("Result", eft_full_grey)
# cv2.waitKey()
# cv2.destroyAllWindows()
