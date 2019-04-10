import numpy as np
import cv2
import sys
import math
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import rosbag
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('/home/chris/ros_workspace/src/video_stabilizer_node/data/youtube_test.mp4')
fgbg = cv2.createBackgroundSubtractorKNN()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

class CarTracker:
    def __init__(self, dt, ID, position_x, position_y):

        self.p_x = KalmanFilter(dim_x=3, dim_z=1)
        self.p_y = KalmanFilter(dim_x=3, dim_z=1)
        self.p_x.F = np.array([[1., dt, 0.5 * dt * dt], [0., 1., dt], [0., 0., 1.]])
        self.p_y.F = np.array([[1., dt, 0.5 * dt * dt], [0., 1., dt], [0., 0., 1.]])

        self.p_x.H = np.array([[1, 0., 0.]])
        self.p_y.H = np.array([[1, 0., 0.]])


        self.R_x_std = 0.00001  # update to the correct value
        self.Q_x_std = 0.7  # update to the correct value
        self.R_y_std = 0.00001  # update to the correct value
        self.Q_y_std = 0.7  # update to the correct value

        self.p_y.Q = Q_discrete_white_noise(dim=3, dt=dt, var=self.Q_y_std ** 2)
        self.p_x.Q = Q_discrete_white_noise(dim=3, dt=dt, var=self.Q_x_std ** 2)

        self.p_x.R *= self.R_x_std ** 2
        self.dt = dt
        self.ID = ID
        self.p_x.x = np.array([[position_x], [0.], [0.]])
        self.p_y.x = np.array([[position_y], [0.], [0.]])
        self.p_x.P *= 100. # can very likely be set to 100.
        self.p_y.P *= 100. # can very likely be set to 100.

        self.time_since_last_update = 0.0

        self.p_y.R *= self.R_y_std ** 2
    def update_pose(self,position_x, position_y):
        self.time_since_last_update = 0.0 # reset time since last update
        self.p_x.update([[position_x]])
        self.p_y.update([[position_y]])
    def predict_pose(self):
        self.time_since_last_update += self.dt #update timer with prediction
        self.p_x.predict()
        self.p_y.predict()
    def get_last_update_time(self):
        return self.time_since_last_update
    def get_position(self):
        return [self.p_x.x[0], self.p_y.x[0]]
    def get_current_error(self):
        return [(self.p_x.P[0])[0], (self.p_y.P[0])[0]]
    def get_total_speed(self):
        # calculate magnitude of speed
        return np.sqrt(self.p_x.x[1]**2+self.p_y.x[1]**2)
    def get_ID(self):
        return self.ID
class CarLocatorNode():
    dt = 1./18.0 # update to the correct value
    got_measurement = False
    first_run = True
    prediction_time = 0.0

    def __init__(self):
        rospy.init_node('single_car_locator')
        self.sub_mes = rospy.Subscriber("/car_pose_measurements", PointStamped, self.measurement_callback) # change to your message type!
        self.pub_update = rospy.Publisher('/kalman_filter/update_pose', PoseWithCovarianceStamped, queue_size=5) # change to your message type!
        self.pub_predict = rospy.Publisher('/kalman_filter/predict_pose', PoseWithCovarianceStamped, queue_size=5) # change to your message type!
        self.pose_msg = PoseWithCovarianceStamped()
        #I have chosen to run this 10 times faster but  2x measurement rate would also work
        rospy.Timer(rospy.Duration(self.dt*10.), self.timer_callback)
        rospy.spin()
    def timer_callback(self, event):
        print 'Timer called at ' + str(rospy.get_time()) #remove when running live with real data
        if(not self.first_run):
            # either we have received a measurement or more than 1.5 sample period has passed
            if(self.got_measurement or ((rospy.get_time()-self.prediction_time)>(1.5*self.dt))):
                print "predict_pose()"  #remove when running live with real data
                self.ct.predict_pose() # might need a mutex
                self.got_measurement = False #clear from last measurement
                self.prediction_time = rospy.get_time()
                self.pose_msg.header = self.pose_msg.header # update to time in the future (add correct code !!!)

                # publishing part of the code
                self.pose_msg.header.frame_id = str(ct.get_ID())
                [self.pose_msg.pose.pose.position.x, self.pose_msg.pose.pose.position.y] = self.ct.get_position()
                [cor_x, cor_y] = self.ct.get_current_error()
                self.pose_msg.pose.covariance[0] = cor_x
                self.pose_msg.pose.covariance[3] = cor_y
                self.pub_predict.publish(self.pose_msg)
    def measurement_callback(self, msg):
        self.pose_msg.header = msg.header # copy the time
        self.got_measurement = True
        if(self.first_run):
            self.ct = CarTracker(self.dt,1,msg.point.x,msg.point.y) # since we have a single car the ID is set to 1
            self.first_run = False # we have received the first measurement
            print "INIT tracker"  #remove when running live with real data
        else:
            self.ct.update_pose(msg.point.x,msg.point.y) # might need a mutex (same as above)
            print "update_pose()"  #remove when running live with real data

        # publishing part of the code
        self.pose_msg.header.frame_id = str(ct.get_ID())
        [self.pose_msg.pose.pose.position.x, self.pose_msg.pose.pose.position.y] = self.ct.get_position()
        [cor_x, cor_y] = self.ct.get_current_error()
        self.pose_msg.pose.covariance[0] = cor_x
        self.pose_msg.pose.covariance[3] = cor_y
        self.pub_update.publish(self.pose_msg)
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
class various_functions:
    def __init__(self):
        self.Mouse_array = []
    def import_file(self, filename):
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
    def export_file(self, filename, array):
        f = open(filename, 'w')
        for x_y in array:
            f.write(str(x_y))
        f.close()
    def on_mouse(self, event, x, y, flag, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)
            pass
            Mouse_array.append(str(x) + " " + str(y) + " ")
    def calc_std_dev(self, array1, array2):
        if (len(array1) != len(array2)):
            "Array lengths do not match!"
            return -1
        std = 0
        for i in range(len(array1)):
            std += math.sqrt((array1[i][0] - array2[i][0]) ** 2 + (array1[i][1] - array2[i][1]) ** 2)
        std = std / len(array1)
        return std
    def correct_to_x_y(self, array):
        temp_array = []
        for i in range(0, len(array), 2):
            temp_array.append((array[i], array[i + 1]))
        return temp_array
    def showImages(self, windowName, image):
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        cv2.imshow(windowName, image)
    def get_warp(self, correct_to, correct_from, image):
        #query_pts = np.float32([kp_image[m.queryIdx].pt for m in correct_to]).reshape(-1, 1, 2)
        #train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in correct_from]).reshape(-1, 1, 2)
        #cv2.UMat(np.array([],
        #M = cv2.getPerspectiveTransform(correct_from, correct_to)
        h, status = cv2.findHomography(correct_to, correct_from)
        img = cv2.warpPerspective(image, h, (1920,1080))
        #img = cv2.warpPerspective(image, M, (1920,1080))
        return img
class receiver:
    def __init__(self):
        self.stabilizer = VideoStabilizer()
        self.mask = cv2.imread("/home/chris/ros_workspace/src/mandatory_2/src/mask1.png")
        self.cap = cv2.VideoCapture('/home/chris/ros_workspace/src/video_stabilizer_node/data/youtube_test.mp4')


        #create left lane init box
    def create_init_boxes(self, image, point1, point2, point3, point4):
        topBox = image.copy()
        mask = np.zeros_like(image)
        cv2.rectangle(mask, point1, point2, (255, 255, 255), -1)
        #cv2.rectangle(mask, point3, point4, (255, 255, 255), -1)
        topBox = cv2.bitwise_and(topBox, mask)
        return topBox
    def stabilize_image(self, image, crop, mask, point1, point2):
        #Stabilize and crop
        frame = self.stabilizer.stabilize_frame(image)
        if mask != 0:
            frame = cv2.bitwise_and(frame, self.mask)
        if crop != 0:
            frame = frame[point1[0]:point1[1], point2[0]:point2[1]]
        return frame
    def remove_background(self, mask):
        #Takes in BW, removes the background
        fgmask = fgbg.apply(mask)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        return fgmask
class LK:
    def __init__(self):
        self.lk_params = dict(winSize=(5, 5),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.color = np.random.randint(0, 255, (100, 3))

        # Parameters for lucas kanade optical flow
        self.p0 = np.array([[[0, 0]]], dtype=np.float32)
        self.old_gray = []
        self.mask = []
        self.tracked_cars = []
        self.old_tracked_cars = []
        self.number_of_cars = 0
        self.ID = 0

        self.kalman_list = []
    def track_car(self, carnumber):
        if len(lucas.p0) >= carnumber-1:
            Car_nr_3.append(lucas.p0[carnumber][0])
    def Euclidean_distance(self, point1, point2):
        return math.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
    def initialize_LK_image(self, image):
        self.mask = np.zeros_like(image)
        self.old_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    def LKStep(self, newFrame, manipulation_image):

        #print("Tracked cars : " + str(len(self.tracked_cars)) + "  " + str(self.tracked_cars))
        self.append_cars()
        for i in range (0, len(self.kalman_list)):
            self.kalman_list[i].predict_pose()
            self.kalman_list[i].update_pose(self.tracked_cars[i][0].copy(), self.tracked_cars[i][1].copy())

            x, y = self.kalman_list[i].get_position()
            speed = self.kalman_list[i].get_total_speed()
            cv2.putText(manipulation_image, "car number: " + str(self.kalman_list[i].ID) + " Speed is: " + str(speed),
                        (int(x + 800), int(y + 320)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            image = cv2.circle(manipulation_image, (int(x + 800), int(y + 320)), 5, self.color[i].tolist(), -1)


        # calculate optical flow
        image = newFrame.copy()
        new_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, new_gray, self.p0, None, **self.lk_params)
        # Select good points
        if len(p1[st==1]) == 0:
            return image
        good_new = p1[st == 1]
        good_old = self.p0[st == 1]


        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            self.mask = cv2.line(self.mask, (a, b), (c, d), self.color[i].tolist(), 2)


        self.old_gray = new_gray
        self.p0 = good_new.reshape(-1, 1, 2)

        return image
    def append_cars(self):
        temp_array = []
        for car in self.p0:
            temp_array.append((car[0][0], car[0][1]))
        for i in range (0, len(temp_array)):
            if not self.matches(temp_array[i], self.tracked_cars):
                self.tracked_cars.append(temp_array[i])
                ct = CarTracker(25, self.ID, temp_array[i][0], temp_array[i][1])
                print(temp_array[i][0], temp_array[i][1])
                self.kalman_list.append(ct)
                self.ID += 1
    def matches(self, point, array_to_find_a_match):
        for i in range (0, len(array_to_find_a_match)):
            dist = self.Euclidean_distance(point, array_to_find_a_match[i])
            if dist < 10:
                array_to_find_a_match[i] = point
                return True
        return False
    def add_point(self, point):
        temp_array = np.append(self.p0, [[[point[0], point[1]]]], axis=0)
        self.p0 = np.array(temp_array, dtype=np.float32)
    def check_for_p0_dup(self, new_cars):
        if len(self.p0) > 0:
            for i in range(0, len(new_cars)):
                if not (self.is_point_being_tracked_in_p0(new_cars[i])):
                    self.add_point(new_cars[i])
        else:
            self.add_point(new_cars)
    def is_point_being_tracked_in_p0(self, point):
        for tracked_car in self.p0:
            distance = self.Euclidean_distance(tracked_car[0], point)
            if distance < 20:
                return True
        return False
    def filter_Contours(self, image, areaMin, areaMax):
        contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        filtered_contours = []
        hulls = []
        for i in range(0, np.alen(contours)):
            # print len(contours)
            if (cv2.contourArea(contours[i]) > areaMin and cv2.contourArea(contours[i]) < areaMax):
                filtered_contours.append(contours[i])

        for i in range(0, len(filtered_contours)):
            hull = cv2.convexHull(filtered_contours[i])
            hulls.append(hull)
        #return hulls
    def get_centroids(self, contours):
        fixed_hulls = []
        for i in range(0, np.alen(contours)):
            m1 = cv2.moments(contours[i])
            cX = int(m1["m10"] / m1["m00"])
            cY = int(m1["m01"] / m1["m00"])
            fixed_hulls.append((cX, cY))
        #return fixed_hulls
    def trackCars(self, centroid):
        cars_to_track = []
        if (len(centroid) > 1):
            for i in range(0, len(centroid)):
                spaced = False
                for j in range(0, len(centroid)):
                    if (i != j):
                        distance = self.Euclidean_distance(centroid[i], centroid[j])
                        if distance > 10:
                            spaced = True
                if spaced:
                    cars_to_track.append(centroid[i])
        elif len(centroid) > 0:
            cars_to_track.append(centroid[0])

        if (len(cars_to_track) != 0):
            self.check_for_p0_dup(cars_to_track)


def main(args):
    org_points = cv2.UMat(np.array([[1126,861], [1264,451], [866,342], [321,852]] ,dtype=np.uint16))
    new_points = cv2.UMat(np.array([[1126,861], [1238,528], [866,342], [728,856]] ,dtype=np.uint16))

#Create objects
    vf = various_functions()
    image_manipulator = receiver()
    lucas = LK()

#Find standard deviation of annotated car
    automatically_annotated_car = vf.import_file("/home/chris/ros_workspace/src/mandatory_2/Without_brackets.txt")
    manual_annotated_car = vf.import_file("/home/chris/ros_workspace/src/mandatory_2/Man.txt")
    xy = vf.correct_to_x_y(automatically_annotated_car)
    xy_man = vf.correct_to_x_y(manual_annotated_car)
    standard_deviation = vf.calc_std_dev(xy, xy_man)
    print ("Standard deviation is: " + str(standard_deviation))

#import frame and create a box for oncomming cars
    ret, frame = image_manipulator.cap.read()
    mask = image_manipulator.stabilize_image(frame, 1, 1,  (320,1080), (800,1200))
    topLeft_initbox = image_manipulator.create_init_boxes(mask, (100, 200), (280, 120), (50, 1080), (144, 700))
    lucas.initialize_LK_image(topLeft_initbox)

###############Loopity Loop###############
    while (True):
#read in the image, and stabilize plus crop
        ret, frame = cap.read()
        mask = image_manipulator.stabilize_image(frame, 1, 1, (320,1080), (800,1200))
        org_image = image_manipulator.stabilize_image(frame, 0, 0, (320,1080), (800,1200))
        lucas_image = lucas.LKStep(mask, org_image)
        no_background = image_manipulator.remove_background(mask)

#Create a Top left detector box both color and BW
        topLeft_BW = image_manipulator.create_init_boxes(no_background, (100, 200), (280, 120), (50, 1080), (144, 700))

#Find the contours of the cars
        hulls = lucas.filter_Contours(topLeft_BW, 60, 500)
        cv2.drawContours(lucas_image, hulls, -1, (255,0,0), 1)
        centers_of_hulls = lucas.get_centroids(hulls)
#Track the cars
        lucas.trackCars(centers_of_hulls)

#show various images.
        #vf.showImages("Lucas kanade overlay", lucas_image)
        vf.showImages("Original image", org_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    print("Launching!")
    main(sys.argv)













    # def check_for_duplicate(self, new_cars):
    #
    #     if len(self.init_cars) != 0:
    #         for i in range (0, len(new_cars)):
    #             if not self.is_point_being_tracked(new_cars[i]):
    #                 self.add_point(new_cars[i])
    #                 self.init_cars.append(new_cars[i])
    #                 self.number_of_cars += 1
    #     self.init_cars = new_cars