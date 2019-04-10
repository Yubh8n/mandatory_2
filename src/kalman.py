#!/usr/bin/env python
from __future__ import print_function
import roslib
import sys
import rospy
import cv2
import numpy as np
from mandatory_2.msg import Num, Num_array
from math import sqrt
import filterpy as KalmanFilter
from filterpy.common import Q_discrete_white_noise

roslib.load_manifest('mandatory_2')


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
class data_collector:
    def __init__(self):
        rospy.init_node('kalman', anonymous=True)
        self.cords_sub = rospy.Subscriber("analyzed_image", Num_array, self.callback)
        self.centroids_pub = rospy.Publisher("Kalman_predictions", Num_array, queue_size=10)
        self.tracked_cars = []
        self.first_run = True
        self.tracked_car = []

    def callback(self, data):
        #print("data in kalman collected")
        result = Num_array()
        result.array = data.array
        Car_array = self.create_car_array(result.array)
        if self.first_run:
            self.tracked_cars = Car_array
            self.first_run = False
        else:
            self.find_new_points(Car_array)
        self.publish_kalman_centroids(Car_array)


    def publish_kalman_centroids(self, array):
        centroid_ros_array = Num_array()
        for centroid in array:
            x_y = Num()
            x_y.x = centroid[0]
            x_y.y = centroid[1]
            centroid_ros_array.array.append(x_y)
        self.centroids_pub.publish(centroid_ros_array)

    def find_new_points(self, array):
        new_cars = []
        for point in array:
            if not self.is_point_being_tracked(point, 150):
                new_cars.append(point)

        return new_cars

    def is_point_being_tracked(self, point, min_distance):
        for trackedpoint in self.tracked_cars:
            distance = self.Euclidean_distance(point, trackedpoint)
            if distance < min_distance:
                return True

            return False

    def match_points(self,array1, array2):
        pass
    def Euclidean_distance(self, point1, point2):
        #print(sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2))
        #print(point1)
        #return 1
        return sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)

    def create_car_array(self, array):
        arrayOfXy = []
        for i in range (1,len(array)):
            x_y = Num()
            x_y = array[i]
            arrayOfXy.append((x_y.x, x_y.y))
            #print("X: " + str(arrayOfXy[i][0]) + " Y: " + str(arrayOfXy[i][1]))
        #print("--------------------------")
        return arrayOfXy

def main(args):
    DC = data_collector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
if __name__ == '__main__':
    print("Launching!")
    main(sys.argv)