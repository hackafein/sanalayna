from head_pose_estimation import PnpHeadPoseEstimator
from time import time

import os
import numpy as np
import pdb

import dlib
import cv2
from imutils import face_utils

# import estimate_head_pose

# multiprocessing may not work on Windows and macOS, check OS for safety.

CNN_INPUT_SIZE = 128

app_directory = os.path.dirname(os.path.abspath(__file__))
LANDMARK_FILE = os.path.join(app_directory,'files/shape_predictor_68_face_landmarks.dat')
FACE_POINTS = os.path.join(app_directory,'assets/model.txt')
CAMERA_FILE = None #os.path.join(app_directory,"assets/camera_parameter_correct.pkl")

O = []
N = []

class StreamProcessor(object):
    def __init__(self, sample_frame, alpha=0.05, gamma=0.1):
        # Monitor the framerate at 1s, 5s, 10s intervals.

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(LANDMARK_FILE)
        
        self.shape = 0

        self.lag = 0
        self.last = time()

        self.tvec = np.array([160,120,-300])
        self.rvec = np.array([0,0,0]) # it's the euler angles of the faces, we are in a XYZ system

        self.cam_w,self.cam_h = sample_frame.shape[:2]

        self.poseEstimator=PnpHeadPoseEstimator(self.cam_w,self.cam_h,assets=CAMERA_FILE)
        self.speed = 1
        
        self.alpha = alpha # alpha is the interval of the stream in seconds
        self.gamma = gamma # gamma is the momentum of of the lag

        self.pose = None
        self.pose_smooth = None
        self.is_new = True
        self.history_shape=[0,0]
        self.history_rvec=[0,0]
    def find_stable_pose(self, frame):
        now = time()
        self.last = now
        new_pose = False

        # the lag variable should be smoothed a little bit to ingest large gaps smoothly 
        self.lag = min(((now-self.last)*0.5+self.lag*(1-0.5)),1) 

        if (self.lag*np.random.rand()<self.alpha) : # and self.speed>count :

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # detect faces in the grayscale frame
            rects = self.detector(gray, 0)

            # loop over the face detections
            if len(rects)==0:
                self.is_new = False
            val=0
            for rect in rects[:1]:
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                self.is_new = True
                shapeInc = self.predictor(gray, rect)
                shapeInc = face_utils.shape_to_np(shapeInc)
                
                print( "Face found")
                self.shape = self.shape*0.+shapeInc*1.
                if self.shape is not None and type(self.shape) is not int:
                    sayac=0
                    for (x, y) in self.shape:
                        if val >0:
                            if abs(float(x)-float(self.history_shape[sayac][0]))<3:
                                self.shape[sayac]=self.history_shape[sayac]
                                sayac+=1
                history_shape=self.shape
               
                """
                shapeInc = self.predictor(gray, rect)
                shapeInc = face_utils.shape_to_np(shapeInc)
                
                # Smooth the facial landmarks
                momentum = max(self.alpha,1-(0.99)**(time()-now))
                self.shape = shapeInc #self.shape*(1-momentum) + shapeInc*momentum
                """
                # Find the head pose
                """
                rvecInc, tvecInc = self.poseEstimator.return_roll_pitch_yaw(
                    self.shape,
                    self.cam_w,
                    self.cam_h)
                """
                _,tvecInc,tvecCent=self.poseEstimator.return_roll_pitch_yaw(self.shape,self.cam_w,self.cam_h)
                pose = self.poseEstimator.return_roll_pitch_yaw_slow(
                    self.shape,
                    self.cam_w,
                    self.cam_h)

                # tvecInc = self.aT.dot(tvecInc-self.bT.T)
                self.tvec = tvecInc # .T[0]
                tvec = self.poseEstimator.camera_matrix.dot(self.tvec)
                tvec = tvec/tvec[2]
                tvec0 = tvec*0+tvecCent*1
                self.tvec[0] = tvec0[0]
                self.tvec[1] = tvec0[1]

                tvecCent[2] = tvec[2]

                #print(self.tvec)
                #print(self.tvec*0.5 + tvecCent*0.5)

                #self.tvec = self.tvec*0.5 + tvecCent*0.5

                rvecInc = pose.get_rotation_euler_angles()
                self.rvec = rvecInc
                
                if self.rvec is not None and type(self.rvec) is not int:
                    sayac=0
                    for (z) in self.rvec:
                        if val >0:
                            if abs(float(self.rvec[sayac][3])-float(self.history_rvec[sayac][3]))<0.1:
                                self.rvec[sayac][3]=self.history_rvec[sayac][3]
                                
                            if abs(float(self.rvec[sayac][2])-float(self.history_rvec[sayac][2]))<0.01:
                                self.rvec[sayac][2]=self.history_rvec[sayac][2]
                            if abs(float(self.rvec[sayac][1])-float(self.history_rvec[sayac][1]))<0.01:
                                self.rvec[sayac][1]=self.history_rvec[sayac][1]
                            sayac+=1 
                history_rvec=self.rvec
                val+=1

                self.pose = np.array([self.rvec,self.tvec])
            
            if self.pose_smooth is not None:
                # smoothing pose
                self.pose_smooth = self.pose_smooth*(1-self.lag)+self.pose*self.lag
            elif self.pose is not None:
                self.pose_smooth = self.pose

            return self.pose_smooth

        else:
            self.is_new = False
            return None

    def draw_boxes(self,frame):
        pose = self.find_stable_pose(frame)
        #if pose is not None:
        #return self.draw_pose(frame,pose)

        return frame


    def draw_shapes(self,frame):
        new_frame = self.draw_boxes(frame)
        if new_frame is not None and self.shape is not None and type(self.shape) is not int:
            for (x, y) in self.shape:
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

        return frame
    
    def returnshapes(self):
        return self.shape

    def draw_pose(self,frame,pose):
        self.pose_estimator.draw_annotation_box(
                frame, pose[0], pose[1])

        return frame

    def get_last_pose(self):
        return self.pose, self.is_new

def main():
    # Video source from webcam or video file.
    video_src = 0
    cam = cv2.VideoCapture(video_src)
    _, sample_frame = cam.read()

    if sample_frame is None:
        exit(-1)

    # Initialize stream processor
    stream = StreamProcessor(sample_frame)
    # new_stream = estimate_head_pose.StreamProcessor(sample_frame)
    history_shape=[0,0]
    val=0
    while True:
        # Read frame, crop it, flip it, suits your needs.
        frame_got, frame = cam.read()
        if frame_got is False:
            break

        # Crop it if frame is larger than expected.
        # frame = frame[0:480, 300:940]

        # If frame comes from webcam, flip it so it looks like a mirror.
        if video_src == 0:
            frame = cv2.flip(frame, 2)

        frame = cv2.imread("tmp.png")
       
        stream.get_last_pose()
        if stream.shape is not None and type(stream.shape) is not int:
            sayac=0
            for (x, y) in stream.shape:
                if val >2:
                    if abs(float(x)-float(history_shape[sayac][0]))<3:
                        stream.shape[sayac]=history_shape[sayac]
                        sayac+=1
        if stream.shape is not None and type(stream.shape) is not int:
            sayac=0
            for (x, y) in stream.shape:
                if sayac==0:
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), 3)
                    solx1=x
                    soly1=y
                if sayac==16:
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)
                    sagx2=x
                    sagy2=y
                else:
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

                sayac+=1
        history_shape=stream.shape



            #pixelratio=0.1
            #yuzgenisligi=(sagx2-solx1)*pixelratio
            #font = cv2.FONT_HERSHEY_SIMPLEX
            
        #    cv2.putText(frame,"YuzMesafe= "+str(yuzgenisligi),(10,100), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        



        frame_old = stream.draw_shapes(frame.copy())
        val+=1
        #pose_old = stream.get_last_pose()
        #print(pose_old,"pose old")
        #frame_new = new_stream.draw_boxes(frame.copy())
        #pose_new = new_stream.get_last_pose()
        #O.append(pose_old)
        #N.append(pose_new)

        #print("OLD =\t{}\n".format(pose_old))

        # Show preview.
        cv2.imshow("Preview", frame)
        if cv2.waitKey(10) == 113:
            cam.release()
            # new_stream.terminate()
            break

if __name__ == '__main__':
    main()
