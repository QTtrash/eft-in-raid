import cv2
import numpy
from matplotlib import pyplot as plt

eft_full = cv2.imread("./images/negative/eft_full.png")
eft_menu = cv2.imread("./images/positive/eft_menu.png")

result = cv2.matchTemplate(eft_full, eft_menu, cv2.TM_CCOEFF_NORMED)
minVal, max_val, minLoc, maxLoc = cv2.minMaxLoc(result)
print(max_val)

# cv2.waitKey()
# cv2.destroyAllWindows()
