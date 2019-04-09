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
from mandatory_2.msg import Num, Num_array
import rospkg

fgbg = cv2.createBackgroundSubtractorKNN()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

rospack = rospkg.RosPack()

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
        self.kalman_sub = rospy.Subscriber("Kalman_predictions", Num_array, self.kalman_callback)
        self.previmg = 0
        self.first_run = True
        self.path = rospack.get_path("mandatory_2")
        self.mask = cv2.imread(self.path + "/src/mask.png")
        self.cars = []
        self.prediction = []
        self.image = []

    def kalman_callback(self, data):
        #Convert ros data to python data.
        kalman_predictions = Num_array()
        kalman_predictions.array = data.array
        for centroids in kalman_predictions.array:
            if len(kalman_predictions.array) > 0:
                self.prediction.append((centroids.x, centroids.y))
                #Draw circles at the kalman predictions
                #cv2.circle(self.image, (centroids.x, centroids.y), 4, (0, 0, 255), -1)

        #Print the predictions and show the image with the drawn circles
        print (self.prediction)
        print("--------------")
        #self.showImage("Kalman image", self.image)
        self.prediction = []

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        cv_image = self.analyze_image(cv_image)

        cropped_image = self.crop_image(cv_image, 0, 1000, 758, 1188)

        subtracted = self.remove_background(cropped_image)
        contours, hierarchy = cv2.findContours(subtracted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        subtracted = cv2.merge((subtracted, subtracted, subtracted))
        subtracted = cv2.bitwise_and(subtracted, cropped_image)

        marked_image = self.mark_cars(cropped_image, contours)
        fgmask = np.hstack((subtracted, marked_image))

        self.showImage("Marked and Sectioned", fgmask)

    def analyze_image(self, image):
        frame = self.stabilizer.stabilize_frame(image)
        frame = cv2.bitwise_and(frame, self.mask)
        return frame

    def crop_image(self, image, xmin, xmax, ymin, ymax):
        frame = image[xmin:xmax, ymin:ymax]
        return frame

    def remove_background(self, mask):
        fgmask = fgbg.apply(mask)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        return fgmask

    def mark_cars(self, image, contours):
        x_y_array = Num_array()
        self.image = image
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

    def opticalFlow(self, current_image, prev_image):
        next = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        prvs = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        self.hsv[..., 0] = ang * 180 / np.pi / 2
        self.hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)

        self.showImage("optical_flow",bgr)

        '''if self.first_run:
            self.first_run = False
            self.previmg = warp
            self.hsv = np.zeros_like(warp)
            self.hsv[..., 1] = 255
        else:
            self.opticalFlow(warp, self.previmg)
            self.previmg = warp'''

    def showImage(self, window_name, image):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
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
