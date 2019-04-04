import numpy as np
import cv2
import sys
import math

cap = cv2.VideoCapture('/home/chris/ros_workspace/src/video_stabilizer_node/data/youtube_test.mp4')
fgbg = cv2.createBackgroundSubtractorKNN()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

#array = []
def on_mouse(event, x, y, flag, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        pass
        #array.append(str(x) + " " + str(y) + " ")

def export_file(filename, array):
    f = open(filename, 'w')
    for x_y in array:
        f.write(str(x_y))
    f.close()

def import_file(filename):
    a = []
    inp = open(filename,"r")
    for line in inp.readlines():
        for i in line.split():
            a.append(int(i))

    temp_array = []
    for i in range(0,len(a)):
        if i % 2 == 0:
            temp_array.append(a[i])
        else:
            temp_array.append(a[i])
    return temp_array

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


#array = import_file("/home/chris/epic.txt")


class receiver:
    def __init__(self):

        self.stabilizer = VideoStabilizer()
        self.first_run = True
        self.cars = []
        self.prediction = []
        self.image = []
        self.mask = cv2.imread("/home/chris/ros_workspace/src/mandatory_2/src/mask.png")
        self.left_lane = []
        self.carList = []


    def Euclidean_distance(self, point1, point2):
        return math.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)

    def set_image(self, image):
        self.image = image

    def set_marker(self, image):
        self.left_lane = image #[430:460, 1060:1090]
        cv2.rectangle(self.left_lane, (0, 475), (1157, 1080), (0, 0, 0), -1)
        cv2.rectangle(self.left_lane, (1027, 436), (1900, 0), (0, 0, 0), -1)
        cv2.rectangle(self.left_lane, (1090, 430), (1125, 491), (0, 0, 0), -1)
    def get_image(self):
        return self.image

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
        for i in range(0, np.alen(contours)):
            if cv2.contourArea(contours[i]) > 50:
                m1 = cv2.moments(contours[i])
                cX = int(m1["m10"] / m1["m00"])
                cY = int(m1["m01"] / m1["m00"])
                cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
                cv2.putText(image, "X: " + str(cX) + " Y: " + str(cY), (cX - 20, cY - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return image

    def showImage(self, window_name, image):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, image)
        cv2.waitKey(1)

    def checkForNewCars(self, x_y):
        for trackedCar in self.carList:
            distance = self.Euclidean_distance(trackedCar, x_y)
            if distance > 50:
                trackedCar.append(x_y)


def main(args):
    counter = 0
    delay = 0
    ic = receiver()
    array = import_file("/home/chris/epic.txt")
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        ic.set_image(frame)
        img_stabilized = ic.analyze_image(frame)
        ic.set_marker(img_stabilized)

        #img_NoBackground = ic.remove_background(img_stabilized)
        img_NoBackground = ic.remove_background(ic.left_lane)


        # Our operations on the frame come here
        # Display the resulting frame
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("frame", 1080,680)
        cv2.setMouseCallback("frame", on_mouse, array)
        if delay > 4:
            contours, hierarchy = cv2.findContours(img_NoBackground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            frame = ic.mark_cars(img_stabilized, contours)

        #cv2.circle(img_stabilized, (array[counter], array[counter + 1]), 2, (255), -1)
        counter += 2
        delay += 1
        cv2.imshow('frame', img_NoBackground)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()
    #export_file("/home/chris/epic.txt", array)


if __name__ == '__main__':
    print("Launching!")
    main(sys.argv)
