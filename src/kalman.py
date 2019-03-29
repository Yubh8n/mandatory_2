#!/usr/bin/env python
from __future__ import print_function
import roslib
import sys
import rospy
import cv2
import numpy as np
import rosbag as bag
from std_msgs.msg import Int32
from mandatory_2.msg import Num, Num_array, Num_2Darray
from mandatory_2.msg import Car_values
from filterpy import kalman

roslib.load_manifest('mandatory_2')

result = Num_array()
x_y = Num()
class kalman_filter:
    def __init__(self):
        rospy.init_node('kalman', anonymous=True)
        self.cords_sub = rospy.Subscriber("analyzed_image", Num_array, self.callback)


    def callback(self, data):
        result.array = data.array
        for i in range (len(result.array)):
            x_y = Num()
            x_y = result.array[i]
            print(x_y.x)



        pass
        #rospy.loginfo("X: " + str(result.x) + " Y: " + str(result.y))


def main(args):
    kf = kalman_filter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("Launching!")
    main(sys.argv)