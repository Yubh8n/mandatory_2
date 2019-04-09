#!/usr/bin/env python
import numpy as np
import rospy
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import PointStamped

class CarTracker:
    ID = 0

    dt = 1.0/15.0
    R_x_std = 0.047 # update to the correct value
    Q_x_std = 0.35 # update to the correct value
    R_y_std = 0.052 # update to the correct value
    Q_y_std = 0.42 # update to the correct value

    time_since_last_update = 0.0

    p_x = KalmanFilter (dim_x=3, dim_z=1)
    p_y = KalmanFilter (dim_x=3, dim_z=1)
    p_x.F =np.array([[1., dt, 0.5*dt*dt],[0., 1., dt],[0., 0., 1.]])
    p_y.F = np.array([[1., dt, 0.5*dt*dt],[0., 1., dt],[0., 0., 1.]])

    p_x.H = np.array([[1.,0.,0.]])
    p_y.H = np.array([[1.,0.,0.]])

    p_x.Q = Q_discrete_white_noise(dim=3, dt=dt, var=Q_x_std**2)
    p_y.Q = Q_discrete_white_noise(dim=3, dt=dt, var=Q_y_std**2)

    p_x.R *= R_x_std**2
    p_y.R *= R_y_std**2

    def __init__(self, dt, ID, position_x, position_y):
        self.dt = dt
        self.ID = ID
        self.p_x.x = np.array([[position_x], [0.], [0.]])
        self.p_y.x = np.array([[position_y], [0.], [0.]])
        self.p_x.P *= 1000. # can very likely be set to 100.
        self.p_y.P *= 1000. # can very likely be set to 100.
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
        return np.sqrt(self.p_x.x[1]**2+self.p_y.x[1]**2) #sqrt(v_x²+v_y²)
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

if __name__ == "__main__":
    node = CarLocatorNode()