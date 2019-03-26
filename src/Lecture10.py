import numpy as np
import cv2


#Exercise 1
#a = np.array([[3,7],[-5,2]])
#b = np.array([3,2])

#x = np.linalg.solve(a,b)

#Exercise 2
img_1 = [230,1781]
img_2 = [2967,1297]
img_3 = [2941,607]
img_4 = [203,59]


obj_1 = [100,1600]
obj_2 = [2900,1600]
obj_3 = [2900,100]
obj_4 = [100,100]

similarity_imgs = [img_1, img_2]
similarity_objs = [obj_1, obj_2]
homo_imgs = [img_1, img_2, img_3, img_4]
homo_objs = [obj_1, obj_2, obj_3, obj_4]
img = cv2.imread("blackboard.jpg")

bb_img1 = [715,379]
bb_img2 = [12,3769]
bb_img3 = [2931,181]
bb_img4 = [2992,4064]

bb_obj1 = [0,0]
bb_obj2 = [0,4160]
bb_obj3 = [3120,0]
bb_obj4 = [3120,4160]

cv2.line(img, (bb_img1[1],bb_img1[0]), (bb_img2[1],bb_img2[0]), 0, 2)
cv2.line(img, (bb_img1[1],bb_img1[0]), (bb_img3[1],bb_img3[0]), 0, 2)
cv2.line(img, (bb_img4[1],bb_img4[0]), (bb_img3[1],bb_img3[0]), 0, 2)
cv2.line(img, (bb_img4[1],bb_img4[0]), (bb_img2[1],bb_img2[0]), 0, 2)

bb_img = [bb_img1,bb_img2,bb_img3,bb_img4]
bb_obj = [bb_obj1, bb_obj2, bb_obj3, bb_obj4]


def locate_template(img, pnt, size):
    template = img[pnt[0]:pnt[0]+size, pnt[1]:pnt[1]+size]
    result = cv2.matchTemplate(img, template, cv2.TM_SQDIFF)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result, None)
    print "\nMinimum value: " + str(minVal) + "\nMax value: " + str(maxVal) + "\nmin location: " + str(minLoc) + "\nmax location: " + str(maxLoc)

    cv2.imshow("template", template)
    cv2.waitKey()

def sim_compose_a(img_points):
    A = []
    for i in range(np.alen(img_points)):
        A.append(list(img_points[i]))
        A.append(list(reversed(img_points[i])))

#Add zeros and ones.(Translation)
    for i in range(0,np.alen(A)):
        if i % 2 == 0:
            A[i].append(1)
            A[i].append(0)
        else:
            A[i][1] = -A[i][1]
            A[i].append(0)
            A[i].append(1)
    return A


def similarity_transform(img_points,obj_points):
    A = sim_compose_a(img_points)
    b = []
    for i in range (0,np.alen(obj_points)):
        b = b + obj_points[i]
    #print np.alen(A)
    #print np.alen(b)
    #print A
    #print b
    x = np.linalg.solve(A,b)
    print np.dot(A,x)

    #return x


def homo_compose_a(img_points):
    A = []
    for i in range(np.alen(img_points)):
        A.append(list(img_points[i]))
        A.append(list(reversed(img_points[i])))

    # Add zeros and ones. (Translation
    for i in range(0, np.alen(A)):
        if i % 2 == 0:
            A[i].append(1)
            A[i].append(0)
        else:
            A[i][1] = -A[i][1]
            A[i].append(0)
            A[i].append(1)
    return A


def homography_transform(img_points, obj_points):
    H = []
    for i in range(0,4):
        H.append([-img_points[i][1], -img_points[i][0],-1,0,0,0,img_points[i][1]*obj_points[i][1],img_points[i][0]*obj_points[i][1],obj_points[i][1]])
        H.append([0, 0, 0, -img_points[i][1],-img_points[i][0], -1, img_points[i][1]*obj_points[i][0], img_points[i][0]*obj_points[i][0], obj_points[i][0]])
    H.append([0, 0, 0, 0, 0, 0, 0, 0, 1])
    b = [0,0,0,0,0,0,0,0,1]
    x = np.linalg.solve(H,b)
    x = np.reshape(x,(3,3))
    return x


#Exercise 3
exercise3 = similarity_transform(similarity_imgs,similarity_objs)
exercise4 = homography_transform(homo_imgs,homo_objs)

img_1.append(1)

img_transformed = np.dot(exercise4,img_1)
result = [img_transformed[0]/img_transformed[2], img_transformed[1]/img_transformed[2]]

exercise5 = homography_transform(bb_img, bb_obj)

locate_template(img, (2000,2000), 50)
print str(2000-25) + " x, " + str(bb_img1[1]-25) + " y"
cv2.rectangle(img, (2000,2000), (2050,2050),0,2)

width, height, colors = np.shape(img)
warp = cv2.warpPerspective(img, exercise5, (height,width))
cv2.namedWindow("warped image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("warped image", 1200,1200)
cv2.imshow("warped image", img)
cv2.waitKey()
cv2.imshow("warped image", warp)
cv2.waitKey()



print np.shape(img)
