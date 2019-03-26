#!/usr/bin/env python
from __future__ import print_function
import roslib
import sys
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
roslib.load_manifest('mandatory_2')

fgbg = cv2.createBackgroundSubtractorMOG2()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

obj1 = [320,200]
obj2 = [320,400]
obj3 = [1000,0]
obj4 = [1000,350]

img1 = [0,0]
img2 = [0,420]
img3 = [1000,0]
img4 = [1000,420]


bb_img = [obj1, obj2, obj3, obj4]
bb_obj = [img1, img2, img3, img4]



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

        frame = frame[0:1000,758:1188]
        return frame
class receiver:
    def __init__(self):
        rospy.init_node('image_shower', anonymous=True)
        #self.image_pub = rospy.Publisher("analyzed_image", Image, queue_size=10)
        self.stabilizer = VideoStabilizer()

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("image_raw", Image, self.callback)


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
            #self.showImage(cv_image)
        except CvBridgeError as e:
            print(e)

        cv_image = self.analyze_image(cv_image)

        H = self.homography_transform(bb_img, bb_obj)
        width, height, colors = np.shape(cv_image)
        warp = cv2.warpPerspective(cv_image, H, (height,width))


        #self.showImage(warp)
        self.backgroundsubtractor(warp)

    def analyze_image(self, image):
        image = self.stabilizer.stabilize_frame(image)
        return image


    def backgroundsubtractor(self, image):
        rows, cols, channels = image.shape
        mask = np.zeros((rows,cols), dtype=image.dtype)
        mask_3chan = cv2.merge((mask,mask,mask))
        cv2.line(mask_3chan, (100, 1000), (310,500), (255, 255, 255),130)
        cv2.line(mask_3chan, (310, 500), (310,200), (255, 255, 255),130)
        cv2.line(mask_3chan, (310, 200), (350,70), (255, 255, 255),130)
        res = cv2.bitwise_and(image, mask_3chan)
        fgmask = fgbg.apply(res)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = cv2.drawContours(fgmask, contours, -1, (0,0,255),2)
        #res1 = cv2.merge((res, res, res))
        #res = cv2.bitwise_and(image, res1)
        #results = np.hstack((fgmask,image))
        self.showImage(fgmask)

    def showImage(self, image):
        cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image window", (1188-758,600))
        cv2.imshow("Image window", image)
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

