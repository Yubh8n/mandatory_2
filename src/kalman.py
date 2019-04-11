#!/usr/bin/env python
from __future__ import print_function
import roslib
import sys
import rospy
import cv2
import numpy as np
from mandatory_2.msg import Num, Num_array, Kalman_feedback, Kalman_feedback_array
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import math

roslib.load_manifest('mandatory_2')


class CarTracker:
    def __init__(self, dt, ID, position_x, position_y):

        self.p_x = KalmanFilter(dim_x=3, dim_z=1)
        self.p_y = KalmanFilter(dim_x=3, dim_z=1)
        self.p_x.F = np.array([[1., dt, 0.5 * dt * dt], [0., 1., dt], [0., 0., 1.]])
        self.p_y.F = np.array([[1., dt, 0.5 * dt * dt], [0., 1., dt], [0., 0., 1.]])

        self.p_x.H = np.array([[1, 0., 0.]])
        self.p_y.H = np.array([[1, 0., 0.]])


        self.R_x_std = 0.01  # update to the correct value
        self.Q_x_std = 7      # update to the correct value
        self.R_y_std = 0.01  # update to the correct value
        self.Q_y_std = 7      # update to the correct value

        self.p_y.Q = Q_discrete_white_noise(dim=3, dt=dt, var=self.Q_y_std ** 2)
        self.p_x.Q = Q_discrete_white_noise(dim=3, dt=dt, var=self.Q_x_std ** 2)

        self.p_x.R *= self.R_x_std ** 2
        self.dt = dt
        self.ID = ID
        self.p_x.x = np.array([[position_x], [0.], [0.]])
        self.p_y.x = np.array([[position_y], [0.], [0.]])
        self.p_x.P *= 100.  # can very likely be set to 100.
        self.p_y.P *= 100.  # can very likely be set to 100.

        self.time_since_last_update = 0.0

        self.p_y.R *= self.R_y_std ** 2

    def update_pose(self,position_x, position_y):
        self.time_since_last_update = 0.0 # reset time since last update
        self.p_x.update([[position_x]])
        self.p_y.update([[position_y]])

    def predict_pose(self):
        self.time_since_last_update += self.dt
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

class data_collector:
    def __init__(self):
        rospy.init_node('kalman', anonymous=True)
        self.cords_sub = rospy.Subscriber("tracked_cars", Num_array, self.callback)
        self.new_car = rospy.Subscriber("new_car", Num, self.newcar_callback)
        self.centroids_pub = rospy.Publisher("Kalman_predictions", Kalman_feedback_array, queue_size=10)
        self.tracked_cars = []
        self.kalman_list = []
        self.kal_id = 0
        self.dt = 1/30.


    def callback(self, data):
        self.tracked_cars = []
        result = Num_array()
        result.array = data.array
        car_array = self.create_car_array(result.array)
        self.update_kalman(car_array, 10)
        self.publish_kalman_centroids()

    def newcar_callback(self, data):
        car = Num()
        car.x = data.x
        car.y = data.y
        new_kalman_element = CarTracker(self.dt, self.kal_id, car.x, car.y)
        self.kal_id += 1
        self.kalman_list.append(new_kalman_element)

    def publish_kalman_centroids(self):
        centroid_ros_array = Kalman_feedback_array()
        for car in self.kalman_list:
            x_y = Kalman_feedback()
            x_y.x = car.p_x.x[0]
            x_y.y = car.p_y.x[0]
            x_y.speed = car.get_total_speed()
            x_y.id  = car.get_ID()
            centroid_ros_array.array.append(x_y)
        print(len(centroid_ros_array.array))
        self.centroids_pub.publish(centroid_ros_array.array)

    @staticmethod
    def euclidean_distance(point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    @staticmethod
    def create_car_array(array):
        array_of_xy = []
        for i in range (1,len(array)):
            x_y = Num()
            x_y = array[i]
            array_of_xy.append((x_y.x, x_y.y))
            #print("X: " + str(array_of_xy[i][0]) + " Y: " + str(array_of_xy[i][1]))
        #print("--------------------------")
        return array_of_xy

    def update_kalman(self, cars, threshold):
        for i in range(len(self.kalman_list)):
            self.kalman_list[i].predict_pose()

        for i in range(0, len(cars)):
            for j in range(0, len(self.kalman_list)):
                distance = self.euclidean_distance(cars[i], (self.kalman_list[j].p_x.x[0], self.kalman_list[j].p_y.x[0]))
                if distance < threshold:
                    self.kalman_list[j].update_pose(cars[i][0],cars[i][1])




'''distance = 9999
        tresh = threshold
        for i in range(0, len(self.tracked_cars)):
            dist_array = []
            any_below_tresh = False
            lowest_dist = 0
            for j in range(0, len(self.tracked_cars)):
                distance = self.euclidean_distance((self.tracked_cars[i][0], self.tracked_cars[i][1]), point)
                if distance < tresh:
                    any_below_tresh = True
                dist_array.append(distance)
            if any_below_tresh:
                for k in range(1, len(dist_array)):
                    if dist_array[k] < dist_array[k - 1]:
                        lowest_dist = k
                self.tracked_cars[i] = dist_array[lowest_dist]
            if distance < tresh:
                self.tracked_cars[i] = point'''




def main(args):
    DC = data_collector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("Kalman node launching!")
    main(sys.argv)









    '''class CarTracker:
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
        return self.ID'''