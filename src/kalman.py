#!/usr/bin/env python
from __future__ import print_function
import roslib
import sys
import rospy
import cv2
import numpy as np
import rosbag as bag
from std_msgs.msg import Int32

roslib.load_manifest('mandatory_2')


class kalman_filter:
    def __init__(self):
        self.cords_sub = rospy.Subscriber("analyzed_image", Int32, self.callback)
        pass



    def self_callback(self, data):
        pass

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