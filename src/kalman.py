#!/usr/bin/env python
from __future__ import print_function
import roslib
import sys
import rospy
import cv2
import numpy as np
from mandatory_2.msg import Num, Num_array
from math import sqrt

roslib.load_manifest('mandatory_2')



class data_collector:
    def __init__(self):
        rospy.init_node('kalman', anonymous=True)
        self.cords_sub = rospy.Subscriber("analyzed_image", Num_array, self.callback)
        self.centroids_pub = rospy.Publisher("Kalman_predictions", Num_array, queue_size=10)
        self.tracked_cars = []
        self.first_run = True
        self.tracked_car = []

    def callback(self, data):
        result = Num_array()
        result.array = data.array
        Car_array = self.create_car_array(result.array)
        if self.first_run:
            self.tracked_cars = Car_array
            self.first_run = False
        else:
            self.find_new_points(Car_array)
        self.publish_kalman_centroids(Car_array)
        #print(Car_array)
        #print("--------------------")

    def kalman(self, point):
        pass

    def PlotX_y(self, array):
        pass


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