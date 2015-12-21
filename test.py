import numpy as np
import cv2
import sys

img = cv2.imread('data/%s'%(sys.argv[1]))
b,g,r = cv2.split(img)
rows, cols = b.shape
T = 170
D = 20
for i in range(rows):
    for j in range(cols):
        if b[i][j] < T:
            b[i][j] = 0
        else:
            b[i][j] = 255
        if g[i][j] < T:
            g[i][j] = 0
        else:
            g[i][j] = 255
        if r[i][j] < T:
            r[i][j] = 0
        else:
            r[i][j] = 255
    
#np.savetxt('test_1_b', b, fmt='%d')

img2 = cv2.merge([b, g, r])
cv2.imshow('test', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
