import numpy as np
import cv2
import sys

cap = cv2.VideoCapture('/home/chris/ros_workspace/src/video_stabilizer_node/data/youtube_test.mp4')
fgbg = cv2.createBackgroundSubtractorKNN()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

array = []

def on_mouse(event, x, y, flag, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        pass
        array.append(str((x, y)) + "\n")

def export_file(filename, array):
    f = open(filename, 'w')
    for x_y in array:
        f.write(str(x_y))
    f.close()

def import_file(filename):
    with open(filename, 'r') as f:

        line = f.read()
        q = line.split(' ')
        a = map(int, q)

    array = []
    #print (a)
    for i in range(0,len(a)):
        if i % 2 == 0:
            array.append(a[i])
        else:
            array.append(a[i])

    #print(array)
    return array

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


array = import_file("/home/chris/epic.txt")


class receiver:
    def __init__(self):

        self.stabilizer = VideoStabilizer()
        self.first_run = True
        self.mask = cv2.imread(self.path + "/src/mask.png")
        self.cars = []
        self.prediction = []
        self.image = []

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
            if cv2.contourArea(contours[i]) > 100:
                m1 = cv2.moments(contours[i])
                cX = int(m1["m10"] / m1["m00"])
                cY = int(m1["m01"] / m1["m00"])
                cv2.circle(image, (cX, cY), 7, (0, 0, 255), -1)
                cv2.putText(image, "X: " + str(cX) + " Y: " + str(cY), (cX - 20, cY - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return image

    def showImage(self, window_name, image):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, image)
        cv2.waitKey(1)


def main(args):
    counter = 0
    ic = receiver()
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Display the resulting frame
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("frame", 1080,680)
        # cv2.setMouseCallback("frame", on_mouse)
        cv2.circle(gray, (array[counter], array[counter + 1]), 2, (255), -1)
        cv2.imshow('frame', gray)
        # print (array[counter][0])
        # print("New_image")
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        counter += 2
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("Launching!")
    main(sys.argv)


cap.release()
cv2.destroyAllWindows()