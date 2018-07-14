# Face-Detection-And-Tracking
Face Detection and tracking using CamShift, Kalman Filter, Optical Flow 

Objective :

1. Detect the face in the first frame of the movie

    Using pre-trained Viola-Jones detector

2. Track the face throughout the movie using:

    CAMShift

    Particle Filter

    Face detector + Kalman Filter (always run the kf.predict(), and run kf.correct() when you get a new face detection)

 
Face Detector + Optical Flow tracker (use the OF tracker whenever the face detector fails).