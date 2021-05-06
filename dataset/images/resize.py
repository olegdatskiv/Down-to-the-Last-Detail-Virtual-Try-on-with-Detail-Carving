import cv2

img = cv2.imread('1983987800_2_3_3.jpg', cv2.IMREAD_UNCHANGED)

print('Original Dimensions : ', img.shape)

width = int(192)
height = int(256)
dim = (width, height)

# resize image
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

print('Resized Dimensions : ', resized.shape)

cv2.imwrite('1983987800_2_3_3.jpg', resized)

cv2.imshow("Resized image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()