#!/usr/bin/env python
from __future__ import print_function
import roslib
import sys
import rospy
import cv2
import numpy as np
import rosbag as bag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Int8
roslib.load_manifest('mandatory_2')
from mandatory_2.msg import Num, Num_array
from mandatory_2.msg import Car_values
import math
import rospkg


fgbg = cv2.createBackgroundSubtractorKNN()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

obj1 = [320, 200]
obj2 = [320, 400]
obj3 = [1000, 0]
obj4 = [1000, 350]

img1 = [0, 0]
img2 = [0, 420]
img3 = [1000, 0]
img4 = [1000, 420]
video = bag.Bag('test.bag', 'w')

bb_img = [obj1, obj2, obj3, obj4]
bb_obj = [img1, img2, img3, img4]

rospack = rospkg.RosPack()

a = Car_values()
#x_y = Num()
#x_y_array = Num_array()
#x_y_2Darray = Num_2Darray()

class VideoStabilizer():
    def __init__(self):
        self.has_seen_first_frame = False

        # Initiate STAR detector
        self.orb = cv2.ORB_create()

        # Initiate FAST object with default values
        self.fast = cv2.FastFeatureDetector_create(nonmaxSuppression=True,
                                                   threshold=100)

        # create BFMatcher object
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def stabilize_frame(self, frame):
        if not self.has_seen_first_frame:
            cv2.imwrite("debug.png", frame)
            self.first_frame = frame
            self.kp_first_frame = self.fast.detect(frame, None)
            self.kp_first_frame, self.des_first_frame = self.orb.compute(frame, self.kp_first_frame)
            self.has_seen_first_frame = True

        kp = self.fast.detect(frame, None)
        kp, des = self.orb.compute(frame, kp)

        # Match descriptors.
        matches = self.bf.match(des, self.des_first_frame)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        good_matches = matches[:100]
        good_matches = matches

        src_pts = np.float32([kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([self.kp_first_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS, 1.0)

        frame = cv2.warpPerspective(frame, M, dsize=(frame.shape[1],
                                                     frame.shape[0]))

        return frame

class receiver:
    def __init__(self):
        rospy.init_node('image_shower', anonymous=True)
        self.image_sub = rospy.Subscriber("image_raw", Image, self.callback)  # Image is not the image, but image from sensor_msgs.msgs
        self.stabilizer = VideoStabilizer()
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("analyzed_image", Num_array, queue_size=10)
        self.previmg = 0
        self.first_run = True
        self.path = rospack.get_path("mandatory_2")
        self.mask = cv2.imread(self.path + "/src/mask.png")
        self.cars = []

    def homo_compose_a(self, img_points):
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

    def homography_transform(self, img_points, obj_points):
        H = []
        for i in range(0, 4):
            H.append([-img_points[i][1], -img_points[i][0], -1, 0, 0, 0, img_points[i][1] * obj_points[i][1],
                      img_points[i][0] * obj_points[i][1], obj_points[i][1]])
            H.append([0, 0, 0, -img_points[i][1], -img_points[i][0], -1, img_points[i][1] * obj_points[i][0],
                      img_points[i][0] * obj_points[i][0], obj_points[i][0]])
        H.append([0, 0, 0, 0, 0, 0, 0, 0, 1])
        b = [0, 0, 0, 0, 0, 0, 0, 0, 1]
        x = np.linalg.solve(H, b)
        x = np.reshape(x, (3, 3))
        return x

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        cv_image = self.analyze_image(cv_image)

        H = self.homography_transform(bb_img, bb_obj)
        width, height, colors = np.shape(cv_image)
        warp = cv2.warpPerspective(cv_image, H, (height, width))
        image = self.backgroundsubtractor(cv_image)
        #warp = self.createMask(warp)
        '''if self.first_run:
            self.first_run = False
            self.previmg = warp
            self.hsv = np.zeros_like(warp)
            self.hsv[..., 1] = 255
        else:
            self.opticalFlow(warp, self.previmg)
            self.previmg = warp'''
        self.showImage("org_img", image)
        #self.showImage("mask", self.mask)

    # Stabilize image
    def analyze_image(self, image):
        image = self.stabilizer.stabilize_frame(image)
        frame = cv2.bitwise_and(image, self.mask)
        frame = frame[0:1000, 758:1188]
        return frame

    # Remove the background
    def remove_background(self, mask):
        fgmask = fgbg.apply(mask)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        return fgmask

    # Mark all cars in the original image
    def mark_cars(self, image, contours):
        x_y_array = Num_array()
        for i in range(0, np.alen(contours)):
            if cv2.contourArea(contours[i]) > 100:
                m1 = cv2.moments(contours[i])
                cX = int(m1["m10"] / m1["m00"])
                cY = int(m1["m01"] / m1["m00"])
                cv2.circle(image, (cX, cY), 7, (0, 0, 255), -1)
                cv2.putText(image, "X: " + str(cX) + " Y: " + str(cY), (cX - 20, cY - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                x_y = Num()
                x_y.x = cX
                x_y.y = cY
                x_y_array.array.append(x_y)
                #a.area = cv2.contourArea(contours[i])
        #print(len(x_y_array.array))
        self.image_pub.publish(x_y_array)
        return image

    # Remove the background and mark the original image
    # plot both the masked and background subtracted image together with the original image with marks
    def backgroundsubtractor(self, original_image):
        subtracted = self.remove_background(original_image)
        contours, hierarchy = cv2.findContours(subtracted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        subtracted = cv2.merge((subtracted, subtracted, subtracted))
        subtracted = cv2.bitwise_and(subtracted, original_image)
        marked_image = self.mark_cars(original_image, contours)
        #fgmask = np.hstack((binary_result, marked_image))
        return subtracted
    def opticalFlow(self, current_image, prev_image):
        next = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        prvs = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        self.hsv[..., 0] = ang * 180 / np.pi / 2
        self.hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)

        self.showImage("optical_flow",bgr)
    def showImage(self, window_name, image):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, (1188 - 758, 600))
        cv2.imshow(window_name, image)
        cv2.waitKey(1)

def main(args):
    ic = receiver()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("Launching!")
    main(sys.argv)
