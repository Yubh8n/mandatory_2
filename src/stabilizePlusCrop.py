#!/usr/bin/env python
from __future__ import print_function
import roslib
import sys
import cv2
import numpy as np
import math
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import rospkg
from mandatory_2.msg import Num, Num_array, Kalman_feedback, Kalman_feedback_array

roslib.load_manifest('mandatory_2')

rospack = rospkg.RosPack()
fgbg = cv2.createBackgroundSubtractorKNN()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


class Communication:
    def __init__(self):
        self.vf = VariousFunctions()
        self.image_manipulator = Receiver()
        rospy.init_node('image_shower', anonymous=True)
        self.image_sub = rospy.Subscriber("image_raw", Image,
                                          self.img_callback)  # Image is not the image, but image from sensor_msgs.msgs
        self.kalman_sub = rospy.Subscriber("Kalman_predictions", Kalman_feedback_array, self.kalman_callback)
        self.stabilizer = VideoStabilizer()
        self.bridge = CvBridge()
        self.tracked_cars = rospy.Publisher("tracked_cars", Num_array, queue_size=10)
        self.first_run = True
        self.path = rospack.get_path("mandatory_2")
        self.mask = cv2.imread(self.path + "/src/mask1.png")
        self.org_points = cv2.UMat(np.array([[1126, 861], [1264, 451], [866, 342], [321, 852]], dtype=np.uint16))
        self.new_points = cv2.UMat(np.array([[1126, 861], [1238, 528], [866, 342], [728, 856]], dtype=np.uint16))
        self.org_img = []
        self.h = self.vf.get_warp(self.org_points, self.new_points)
        self.tracker = OwnTracker(self.h)

    def kalman_callback(self, data):
        result = Num_array()
        result.array = data.array
        car_array = self.create_car_array(result.array)

        self.tracker.draw_circles(self.org_image, car_array)
        self.vf.show_images("Marked image", self.org_image)

    def img_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)


        ###############Loopty Loop###############

        # read in the image, and stabilize plus crop
        mask = self.image_manipulator.stabilize_image(cv_image, 1, 1, (320, 1000), (800, 1200))
        self.org_image = self.image_manipulator.stabilize_image(cv_image, 0, 0, (320, 1080), (800, 1200))
        # Remove the background from the image
        no_background_binary = self.image_manipulator.remove_background(mask)

        # Create a Top left detector box in black and white
        init_boxes_binary = self.image_manipulator.create_init_boxes(no_background_binary, (100, 200), (280, 120),
                                                                     (100, 1080), (200, 500))

        # Find the contours of the cars in the new image
        hulls = self.tracker.filter_contours(init_boxes_binary, 30, 2000)
        centers_of_hulls = self.tracker.get_centroids(hulls)
        self.tracker.append_cars(centers_of_hulls)

        # Run the tracking algorithm on the new incomming image
        self.tracked = self.tracker.wannabe_lkstep(no_background_binary, mask)

        if len(self.tracker.tracked_cars) > 1:
            x_y_array = Num_array()
            for car in self.tracker.tracked_cars:
                x_y = Num()
                x_y.x = car[0]
                x_y.y = car[1]
                x_y_array.array.append(x_y)
            self.tracked_cars.publish(x_y_array)

    @staticmethod
    def create_car_array(array):
        arrayOfXy = []
        for i in range(1, len(array)):
            x_y = array[i]

            arrayOfXy.append((x_y.x, x_y.y, x_y.id, x_y.speed))
        return arrayOfXy


class VideoStabilizer:
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


class VariousFunctions:
    def __init__(self):
        pass

    @staticmethod
    def import_file(filename):
        a = []
        inp = open(filename, "r")
        for line in inp.readlines():
            for i in line.split():
                a.append(float(i))

        temp_array = []
        for i in range(0, len(a)):
            if i % 2 == 0:
                temp_array.append(a[i])
            else:
                temp_array.append(a[i])
        return temp_array

    @staticmethod
    def export_file(filename, array):
        f = open(filename, 'w')
        for x_y in array:
            f.write(str(x_y))
        f.close()

    @staticmethod
    def calc_std_dev(array1, array2):
        if (len(array1) != len(array2)):
            "Array lengths do not match!"
            return -1
        std = 0
        for i in range(len(array1)):
            std += math.sqrt((array1[i][0] - array2[i][0]) ** 2 + (array1[i][1] - array2[i][1]) ** 2)
        std = std / len(array1)
        return std

    @staticmethod
    def correct_to_x_y(array):
        temp_array = []
        for i in range(0, len(array), 2):
            temp_array.append((array[i], array[i + 1]))
        return temp_array

    @staticmethod
    def show_images(window_name, image):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, image)
        cv2.waitKey(1)

    @staticmethod
    def get_warp(correct_to, correct_from):
        h, status = cv2.findHomography(correct_to, correct_from)
        #img = cv2.warpPerspective(image, h, (1920, 1080))
        return h


class Receiver:
    def __init__(self):
        self.stabilizer = VideoStabilizer()
        self.path = rospack.get_path("mandatory_2")
        self.mask = cv2.imread(self.path + "/src/mask1.png")
        # self.cap = cv2.VideoCapture('/home/chris/ros_workspace/src/video_stabilizer_node/data/youtube_test.mp4')

        # create left lane init box

    def create_init_boxes(self, image, point1, point2, point3, point4):
        topBox = image.copy()
        mask = np.zeros_like(image)
        cv2.rectangle(mask, point1, point2, (255, 255, 255), -1)
        #cv2.rectangle(mask, point3, point4, (255, 255, 255), -1)
        topBox = cv2.bitwise_and(topBox, mask)
        return topBox

    def stabilize_image(self, image, crop, mask, point1, point2):
        # Stabilize and crop
        frame = self.stabilizer.stabilize_frame(image)
        if mask != 0:
            frame = cv2.bitwise_and(frame, self.mask)
        if crop != 0:
            frame = frame[point1[0]:point1[1], point2[0]:point2[1]]
        return frame

    def remove_background(self, mask):
        # Takes in BW, removes the background
        fgmask = fgbg.apply(mask)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        return fgmask


class OwnTracker:
    def __init__(self, h):
        self.tracked_cars = []
        self.new_car = rospy.Publisher("new_car", Num, queue_size=10)
        self.h = h

    @staticmethod
    def euclidean_distance(point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def wannabe_lkstep(self, masked_image, org_image):
        returnimage = org_image.copy()
        hulls1 = self.filter_contours(masked_image, 50, 2000)
        cv2.drawContours(returnimage, hulls1, -1, (0, 255, 0), 1)
        centers_of_hulls1 = self.get_centroids(hulls1)
        self.update(centers_of_hulls1)
        return returnimage

    @staticmethod
    def draw_circles(image, array_of_x_y_coords):
        for i in range(0, len(array_of_x_y_coords)):
            #if array_of_x_y_coords[i][1] < 600:
            #image = cv2.circle(image, (array_of_x_y_coords[i][0]+800, array_of_x_y_coords[i][1]+320), 3, (0, 0, 255), -1)
            image = cv2.putText(image, "x: " + str(array_of_x_y_coords[i][0]) + " y: " + str(array_of_x_y_coords[i][1]) + " car number: " + str(array_of_x_y_coords[i][2]) + " speed: " + str(array_of_x_y_coords[i][3]),
                                (int(array_of_x_y_coords[i][0]+820), int(array_of_x_y_coords[i][1]+320)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            image = cv2.rectangle(image, (array_of_x_y_coords[i][0]-10+800,array_of_x_y_coords[i][1]-10+320),(array_of_x_y_coords[i][0]+10+800,array_of_x_y_coords[i][1]+10+320), (0,255,0), 2)

    def update(self, array_of_centroids):
        for car in array_of_centroids:
            self.is_point_being_tracked(car, 15)

    def append_cars(self, array):
        temp_array = []
        for car in array:
            temp_array.append((car[0], car[1]))
        for i in range(0, len(temp_array)):
            if not self.matches(temp_array[i], self.tracked_cars):
                self.tracked_cars.append(temp_array[i])
                x_y = Num()
                x_y.x = temp_array[i][0]
                x_y.y = temp_array[i][1]

                self.new_car.publish(x_y)

    def matches(self, point, array_to_find_a_match):
        for i in range(0, len(array_to_find_a_match)):
            dist = self.euclidean_distance(point, array_to_find_a_match[i])
            if dist < 20:
                return True
        return False

    def is_point_being_tracked(self, point, threshold):
        tresh = threshold
        for i in range(0, len(self.tracked_cars)):
            for j in range(0, len(self.tracked_cars)):
                distance = self.euclidean_distance((self.tracked_cars[i][0], self.tracked_cars[i][1]), point)
                if distance < tresh:
                    self.tracked_cars[i] = point

    @staticmethod
    def filter_contours(image, area_min, area_max):
        contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        filtered_contours = []
        hulls = []
        for i in range(0, np.alen(contours)):
            if area_min < cv2.contourArea(contours[i]) < area_max:
                filtered_contours.append(contours[i])

        for i in range(0, len(filtered_contours)):
            hull = cv2.convexHull(filtered_contours[i])
            hulls.append(hull)
        return hulls

    @staticmethod
    def get_centroids(contours):
        fixed_hulls = []
        for i in range(0, np.alen(contours)):
            m1 = cv2.moments(contours[i])
            cX = int(m1["m10"] / m1["m00"])
            cY = int(m1["m01"] / m1["m00"])
            fixed_hulls.append((cX, cY))
        return fixed_hulls


def main(args):
    inter_communication = Communication()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("Launching!")
    main(sys.argv)