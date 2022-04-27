import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('not_cut_properly.jpg', cv2.IMREAD_COLOR)
img1 = img.copy()
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# gray_blurred = cv2.GaussianBlur(gray, (11,11), cv2.BORDER_DEFAULT)
gray_blurred = cv2.blur(gray, (3, 3))

#Hough Transform
detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 10, param1 = 100,
                                    param2 = 100, minRadius = 0, maxRadius = 0)

#Drawing and Printing detected circles
if detected_circles is not None:
  
    detected_circles = np.uint16(np.around(detected_circles))
    print("detected circles", detected_circles)
    for pt in detected_circles[0, :]:
        x, y, r = pt[0], pt[1], pt[2]
        cv2.circle(img1, (x, y), r, (0, 255, 0), 2)
        cv2.circle(img1, (x, y), 1, (0, 0, 255), 3)

#Thresholding
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

#Getting pixels on the circumference of circle
def get_points_on_circle(x_centre, y_centre, radius):
    points=[]
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    cv2.circle(mask, center=(x_centre, y_centre), radius=radius, color=255, thickness=1)
    y,x=np.where(mask)
    points=list(zip(y,x))
    return points

#Checking neighbouring pixels
def check_neighbours(img, points, dist):
    badpixels=[]
    for point in points:
        rows, cols = img.shape
        i, j = point[1], point[0]
        rmin = i - dist if i - dist >= 0 else 0
        rmax = i + dist if i + dist < rows else i
        cmin = j - dist if j - dist >= 0 else 0
        cmax = j + dist if j + dist < cols else j
        neighbours = []

        for x in range(rmin, rmax + 1):
            for y in range(cmin, cmax + 1):
                neighbours.append([x, y])
        neighbours.remove([point[1], point[0]])
        neighbours_flip = np.flip(neighbours)
        
        values=[]
        for px in neighbours_flip:
            y, x = px[0], px[1]
            values.append(thresh[y,x])
        
        if(np.max(values)==0):
           badpixels.append(point)
    
    return badpixels

## Test Run on 1 circle
# img2=img.copy()
# points=get_points_on_circle(496, 64, 34)
# for point in points:
#     if thresh[point]==255:
#         points.remove(point)
# bad_pixels=check_neighbours(thresh,points,1)
# print(bad_pixels)
# if bad_pixels:
#     bad2=np.flip(bad_pixels)
#     for px in bad2:
#         x=px[0]
#         y=px[1]
#         cv2.circle(img2, (x, y), 1, (0, 0, 255), 2)
    
# cv2.imshow("bad pixels", img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

img3=img.copy()
def check_convexity():
    for circle in detected_circles[0, :]:
        x, y, r = circle[0], circle[1], circle[2]
        points = get_points_on_circle(x,y,r)
        # for point in points:
        #     if thresh[point]==255:
        #         points.remove(point)

        bad_pixels=check_neighbours(thresh,points,1)
        if bad_pixels:
            print("bad_pixels", bad_pixels)
            bad_pixels_flip=np.flip(bad_pixels)
            for px in bad_pixels_flip:
                a=px[0]
                b=px[1]
                cv2.circle(img3, (a, b), 1, (0, 0, 255), 2)
        
check_convexity()
cv2.imshow("detected circles", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("thresholding", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("bad pixels", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()   