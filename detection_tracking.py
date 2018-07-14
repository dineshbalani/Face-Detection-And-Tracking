import os
import sys
import cv2
import matplotlib.pyplot as plt
# from pykalman import KalmanFilter
import numpy as np
from math import cos, sin, sqrt


face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

def help_message():
   print("Usage: [Question_Number] [Input_Video] [Output_Directory]")
   print("[Question Number]")
   print("1 Camshift")
   print("2 Particle Filter")
   print("3 Kalman Filter")
   print("4 Optical Flow")
   print("[Input_Video]")
   print("Path to the input video")
   print("[Output_Directory]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "02-1.avi " + "./")

def CamShift(frame,roi_hist,track_window):
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
    ret, track_window = cv2.CamShift(dst, track_window, term_crit)
    return ret
    return pts


def particleevaluator(back_proj, particle):
    return back_proj[particle[1], particle[0]]



def detect_one_face(im):
    gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 3)
    if len(faces) == 0:
        return (0,0,0,0)
    # print(faces)
    return faces[0]

def hsv_histogram_for_window(frame, window):
    # set up the ROI for tracking
    c,r,w,h = window
    roi = frame[r:r+h, c:c+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    return roi_hist


def resample(weights):
    n = len(weights)
    indices = []
    C = [0.] + [sum(weights[:i+1]) for i in range(n)]
    u0, j = np.random.random(), 0
    for u in [(u0+i)/n for i in range(n)]:
      while u > C[j]:
          j+=1
      indices.append(j-1)
    return indices

def OpticalFlow_Tracker(v,file_name):
    output_name = sys.argv[3] + file_name
    output = open(output_name, "w")
    frameCounter = 0
    # read first frame
    ret, old_frame = v.read()
    # print(v)
    if ret == False:
        return

    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    # ret, old_frame = v.read()
    c,r,w,h = detect_one_face(old_frame)
    output.write("%d,%d,%d\n" % (0, c + w / 2, r + h / 2))  # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1
    # print(face)
    # print(old_frame.shape)
    # old_frame_f = old_frame[r:r+h,c:c+w]
    # # print(old_frame.shape)
    # # print(face)
    # # print(old_frame1)
    # cv2.imshow('frame', old_frame_f)
    # k = cv2.waitKey()
    # old_gray_f = cv2.cvtColor(old_frame_f, cv2.COLOR_BGR2GRAY)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    #
    # p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    p0 = [[((c+w/2), (r+h/2))]]
    p0 = np.float32(np.asarray(p0))
    # p0 = face
    # print(p0)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while (1):
        ret, frame = v.read()
        if ret == False:
            break
        # c, r, w, h = detect_one_face(frame)
        # frame = frame[c:c + w, r:r + h]
        # print(frame.shape)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        # print(p1)

        good_new = p1[st == 1]
        good_old = p1[st == 1]

        # draw the tracks
        # for i, (new, old) in enumerate(zip(good_new, good_old)):
        #     a, b = new.ravel()
        #     # print(a,b)
        #     # print(fadfa)
        #     c, d = old.ravel()
        #     mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        #     frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        # img = cv2.add(frame, mask)
        #
        # cv2.imshow('frame', img)
        # k = cv2.waitKey(30) & 0xff
        # if k == 27:
        #     break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = p1
        # print(p0)
        # p0 = good_new.reshape(-1, 1, 2)
        # print(p0)
        output.write("%d,%d,%d\n" % (frameCounter, p1[0][0][0], p1[0][0][1]))  # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1

    cv2.destroyAllWindows()
    v.release()

def Particle_tracker(v,file_name):
    output_name = sys.argv[3] + file_name
    output = open(output_name, "w")
    frameCounter = 0
    # read first frame
    ret, frame = v.read()
    if ret == False:
        return

    # print(frame.shape)
    # detect face in first frame
    c, r, w, h = detect_one_face(frame)
    # print(c, r, w, h)
    # print(c + w / 2, r + h / 2)
    # Write track point for first frame
    output.write("%d,%d,%d\n" % (0, c + w / 2, r + h / 2))  # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1
    # set the initial tracking window
    track_window = (c, r, w, h)
    roi_hist = hsv_histogram_for_window(frame, (c, r, w, h))  # this is provided for you

    hsvt = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    hist_bp = cv2.calcBackProject([hsvt], [0, 1], roi_hist, [0, 180], 1)
    n_particles = 200
    init_pos = np.array([c + w / 2.0, r + h / 2.0], int)  # Initial position
    particles = np.ones((n_particles, 2), int) * init_pos  # Init particles to init position
    f0 = particleevaluator(hist_bp, init_pos) * np.ones(n_particles)  # Evaluate appearance model
    weights = np.ones(n_particles) / n_particles

    # initialize the tracker
    # e.g. kf = cv2.KalmanFilter(4,2,0)
    # or: particles = np.ones((n_particles, 2), int) * initial_pos

    while (1):
        ret, frame = v.read()  # read another frame
        if ret == False:
            break
        stepsize = 13
        hsvt = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        hist_bp = cv2.calcBackProject([hsvt], [0, 1], roi_hist, [0, 180], 1)
        # Particle motion model: uniform step (TODO: find a better motion model)
        np.add(particles, np.random.uniform(-stepsize, stepsize, particles.shape), out=particles, casting="unsafe")
        # initx = np.average(particles, weights=weights, axis=0)
        # particles = create_gaussian_particles(
        #     mean=initx, std=(5, 5, np.pi / 4), N=n_particles)

        # Clip out-of-bounds particles
        particles = particles.clip(np.zeros(2), np.array((frame.shape[1], frame.shape[0])) - 1).astype(int)

        f = particleevaluator(hist_bp, particles.T)  # Evaluate particles
        weights = np.float32(f.clip(1))  # Weight ~ histogram response
        weights /= np.sum(weights)  # Normalize w
        # print(weights)
        pos = np.sum(particles.T * weights, axis=1).astype(int)  # expected position: weighted average

        if 1. / np.sum(weights ** 2) < n_particles / 2.:  # If particle cloud degenerate:
            particles = particles[resample(weights), :]  # Resample particles according to weights\

        # for pt in particles:
        #     img2 = cv2.circle(frame, (int(pt[0]), int(pt[1])), 1, 255, -1)
        # img2 = cv2.circle(frame, (int(pos[0]), int(pos[1])), 3, 55, -1)
        # print(pos)
        # pts = cv2.boxPoints(pos)
        # print(pts)
        # pts = np.int0(pts)
        # img2 = cv2.polylines(frame, [pts], True, 255, 2)

        # cv2.imshow('img2', img2)
        # k = cv2.waitKey(60) & 0xff
        # if k == 27:
        #     break
        # else:
        #     cv2.imwrite(chr(k) + ".jpg", img2)

        # mean = np.average(particles, weights=weights, axis=0)
        # frame = cv2.rectangle(frame, (c, r), (c + w, r + h), 255, 2)
        output.write("%d,%d,%d\n" % (frameCounter,pos[0],pos[1]))# Write as frame_index,pt_x,pt_y

        frameCounter = frameCounter + 1
        # print(pos)
    output.close()

def Kalman_tracker(v,file_name):
    output_name = sys.argv[3] + file_name
    output = open(output_name, "w")

    frameCounter = 0
    # read first frame
    ret, frame = v.read()
    if ret == False:
        return

    # print(frame.shape)
    # detect face in first frame
    c, r, w, h = detect_one_face(frame)
    # Write track point for first frame
    output.write("%d,%d,%d\n" % (0, c + w / 2, r + h / 2))  # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1
    # set the initial tracking window
    track_window = (c, r, w, h)

    # state = np.array([c + w / 2, r + h / 2], dtype='float64')  # initial position
    # kalman = cv2.KalmanFilter(2, 1, 0)  # 4 state/hidden, 2 measurement, 0 control
    # kalman.transitionMatrix = np.array([[1., 0.],  # a rudimentary constant speed model:
    #                                     [0., 1.]])
    # kalman.measurementMatrix = 1. * np.eye(2, 2)  # you can tweak these to make the tracker
    # kalman.processNoiseCov = 1e-10 * np.eye(2, 2)  # respond faster to change and be less smooth
    # kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
    # kalman.errorCovPost = 1e-1 * np.eye(2, 2)
    # kalman.statePost = state
    state = np.array([c + w / 2, r + h / 2, 0, 0], dtype='float64')  # initial position
    kalman = cv2.KalmanFilter(4, 2, 0)  # 4 state/hidden, 2 measurement, 0 control
    kalman.transitionMatrix = np.array([[1., 0., .1, 0.],  # a rudimentary constant speed model:
                                        [0., 1., 0., .1],  # x_t+1 = x_t + v_t
                                        [0., 0., 1., 0.],
                                        [0., 0., 0., 1.]])
    kalman.measurementMatrix = 1. * np.eye(2, 4)  # you can tweak these to make the tracker
    kalman.processNoiseCov = 1e-5 * np.eye(4, 4)  # respond faster to change and be less smooth
    kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
    kalman.errorCovPost = 1e-1 * np.eye(4, 4)
    kalman.statePost = state
    # initialize the tracker
    # e.g. kf = cv2.KalmanFilter(4,2,0)
    # or: particles = np.ones((n_particles, 2), int) * initial_pos
    # count = 0

    while (1):
        ret, frame = v.read()  # read another frame
        if ret == False:
            break
        prediction = kalman.predict()
        # print('p',prediction)
        # print('s',state)
        # obtain measurement
        # measurement = kalman.measurementNoiseCov * np.random.randn(1, 2)
        # print(kalman.measurementNoiseCov)
        # print(np.dot(kalman.measurementMatrix, state))
        c, r, w, h = detect_one_face(frame)
        # print(c,r,w,h)
        # print(c + w / 2)
        measurement = np.array([c+w/2,r+h/2], dtype='float64')
        # measurement = np.dot(kalman.measurementMatrix, state)
        final = prediction
        # print('hi)
        # print('m',measurement[0])
        # print(c+w/2)
        # print(fdknfd)
        # if (c+w/2)== measurement[0] and (r+h/2)==measurement[1]:  # e.g. face found
        # if (c+w/2)-3<= measurement[0] <= (c+w/2)+3 and (r+h/2)-3<=measurement[1] <= (r+h/2)+3:  # e.g. face found
        if c!=0 and w!=0 and r!=0 and h!=0:
            # print('hi')
            # count+=1
            # print(hiii)
            # measurement = np.array([int(measurement[0][0]), int(measurement[0][1])], dtype='float64')
            # measurement = state = np.array([c + w / 2, r + h / 2], dtype='float64')
            # print(dfnkjas)
            posterior = kalman.correct(measurement)
            final = posterior
            # process_noise = sqrt(kalman.processNoiseCov[0, 0]) * np.random.randn(2, 1)
            # print('s', state)

            # print(dfsdjk)

        # state = np.array([c + w / 2, r + h / 2], dtype='float64')  # initial position

        # process_noise = sqrt(kalman.processNoiseCov[0, 0]) * np.random.randn(2, 1)
        # state = np.dot(kalman.transitionMatrix, state)
        # state = np.array([c + w / 2, r + h /2 ,0,0], dtype='float64')  # initial position
        # img2 = cv2.circle(frame, (int(final[0]), int(final[1])), 5, 255, -1)
        # cv2.imshow('img2', img2)
        # k = cv2.waitKey(60) & 0xff
        # if k == 27:
        #     break
        # else:
        #     cv2.imwrite(chr(k) + ".jpg", img2)

        output.write("%d,%d,%d\n" % (frameCounter,final[0],final[1]))# Write as frame_index,pt_x,pt_y

        frameCounter = frameCounter + 1

    # print(coun/t)
    output.close()


def skeleton_tracker(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # print(frame.shape)
    # detect face in first frame
    c,r,w,h = detect_one_face(frame)
    # print(c,r,w,h)
    # Write track point for first frame
    output.write("%d,%d,%d\n" % (0,c + w / 2,r+h/2)) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # set the initial tracking window
    track_window = (c,r,w,h)


    # calculate the HSV histogram in the window
    # NOTE: you do not need this in the Kalman, Particle or OF trackers
    roi_hist = hsv_histogram_for_window(frame, (c,r,w,h)) # this is provided for you

    hsvt = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    hist_bp = cv2.calcBackProject([hsvt], [0, 1], roi_hist, [0, 180, 0, 256], 1)
    n_particles = 200

    init_pos = np.array([c + w / 2.0, r + h / 2.0], int)  # Initial position
    particles = np.ones((n_particles, 2), int) * init_pos  # Init particles to init position
    f0 = particleevaluator(hist_bp, init_pos) * np.ones(n_particles)  # Evaluate appearance model
    weights = np.ones(n_particles) / n_particles  # weights are uniform (at first)

    # initialize the tracker
    # e.g. kf = cv2.KalmanFilter(4,2,0)
    # or: particles = np.ones((n_particles, 2), int) * initial_pos

    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break


        # perform the tracking
        ret = CamShift(frame,roi_hist,track_window)
        # KalmanFilter(c,w,r,h,frame.shape[0],frame.shape[1])
        # ParticleFilter(c,w,r,h,frame,hist_bp,particles,n_particles)
        # e.g. cv2.meanShift, cv2.CamShift, or kalman.predict(), kalman.correct()
        # print(ret)
        # Draw it on image
        pts = cv2.boxPoints(ret)
        # print(pts)
        # pts = np.int0(pts)

        # img2 = cv2.polylines(frame, [pts], True, 255, 2)
        # img2 = cv2.circle(frame,(int(ret[0][0]),int(ret[0][1])),5,255,-1)
        # cv2.imshow('img2', img2)
        # k = cv2.waitKey(60) & 0xff
        # if k == 27:
        #     break
        # else:
        #     cv2.imwrite(chr(k) + ".jpg", img2)

        # print(ret[0])

        # use the tracking result to get the tracking point (pt):
        # if you track a rect (e.g. face detector) take the mid point,
        # if you track particles - take the weighted average
        # the Kalman filter already has the tracking point in the state vector

        # write the result to the output file
        output.write("%d,%d,%d\n" % (frameCounter,ret[0][0],ret[0][1]))# Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1

    output.close()

if __name__ == '__main__':
    question_number = -1
   
    # Validate the input arguments
    if (len(sys.argv) != 4):
        help_message()
        sys.exit()
    else: 
        question_number = int(sys.argv[1])
        if (question_number > 4 or question_number < 1):
            print("Input parameters out of bound ...")
            sys.exit()

    # read video file
    video = cv2.VideoCapture(sys.argv[2]);

    if (question_number == 1):
        skeleton_tracker(video, "output_camshift.txt")
    elif (question_number == 2):
        Particle_tracker(video, "output_particle.txt")
    elif (question_number == 3):
        Kalman_tracker(video, "output_kalman.txt")
    elif (question_number == 4):
        OpticalFlow_Tracker(video, "output_of.txt")


'''
For Kalman Filter:

# --- init
        kalman = cv2.KalmanFilter(2, 1, 0)

state = np.array([c+w/2,r+h/2,0,0], dtype='float64') # initial position
kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                    [0., 1., 0., .1],
                                    [0., 0., 1., 0.],
                                    [0., 0., 0., 1.]])
kalman.measurementMatrix = 1. * np.eye(2, 4)
kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
kalman.errorCovPost = 1e-1 * np.eye(4, 4)
kalman.statePost = state


# --- tracking

prediction = kalman.predict()

# ...
# obtain measurement

if measurement_valid: # e.g. face found
    # ...
    posterior = kalman.correct(measurement)

# use prediction or posterior as your tracking result
'''

'''
For Particle Filter:

# --- init

# a function that, given a particle position, will return the particle's "fitness"
def particleevaluator(back_proj, particle):
    return back_proj[particle[1],particle[0]]

# hist_bp: obtain using cv2.calcBackProject and the HSV histogram
# c,r,w,h: obtain using detect_one_face()
n_particles = 200

init_pos = np.array([c + w/2.0,r + h/2.0], int) # Initial position
particles = np.ones((n_particles, 2), int) * init_pos # Init particles to init position
f0 = particleevaluator(hist_bp, pos) * np.ones(n_particles) # Evaluate appearance model
weights = np.ones(n_particles) / n_particles   # weights are uniform (at first)


# --- tracking

# Particle motion model: uniform step (TODO: find a better motion model)
np.add(particles, np.random.uniform(-stepsize, stepsize, particles.shape), out=particles, casting="unsafe")

# Clip out-of-bounds particles
particles = particles.clip(np.zeros(2), np.array((im_w,im_h))-1).astype(int)

f = particleevaluator(hist_bp, particles.T) # Evaluate particles
weights = np.float32(f.clip(1))             # Weight ~ histogram response
weights /= np.sum(weights)                  # Normalize w
pos = np.sum(particles.T * weights, axis=1).astype(int) # expected position: weighted average

if 1. / np.sum(weights**2) < n_particles / 2.: # If particle cloud degenerate:
    particles = particles[resample(weights),:]  # Resample particles according to weights
# resample() function is provided for you
'''
