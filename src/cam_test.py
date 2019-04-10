from cv2 import *
import roslib
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
roslib.load_manifest('transmit_usb')
from mandatory_2.msg import Num, Num_array
import rospkg
# initialize the camera

class reciever:
    def __init__(self):
        rospy.init_node('image_shower', anonymous=True)
        self.image_sub = rospy.Subscriber("image_raw", Image, self.callback)  # Image is not the image, but image from sensor_msgs.msgs
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("analyzed_image", Image, queue_size=10)




cam = VideoCapture(0)   # 0 -> index of camera
s, img = cam.read()
if s:    # frame captured without any errors
    while True:
        s, img = cam.read()
        #namedWindow("cam-test",cv)
        #imshow("cam-test",img)
        waitKey(1)
        #destroyWindow("cam-test")
        #imwrite("filename.jpg",img) #save image