import numpy as np
import cv2

class KalmanParameters: 
    # Initialize Kalman filter parameters
    kalman = cv2.KalmanFilter(4, 2)  # 4 state variables, 2 measurements
    
    # state vectoru bu matrixle carpinca sadece x ve y konumlarini donduruyor
    #hiz degerleri 0la carpildigindan yok oluyor
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                        [0, 1, 0, 0]], np.float32)
    
    #ilk row x pozisyonu + hız*zaman(1 ms) eklemeye yariyor carpim durumunda
    #ikinci row aynisini y icin yapiyor
    #3.row x hizini oldugu gibi veriyor
    #4.row y hizini oldugu gibi veriyor
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
    
    # Initialize process and measurement noise covariances
    #tune icin katsayi ile oyna
    kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32) * 0.01
    
    #tune icin katsayi ile oyna
    kalman.measurementNoiseCov = np.array([[1, 0],
                                        [0, 1]], np.float32) * 0.01
    
    # Initial state
    kalman.statePre = np.array([[0], [0], [0], [0]], np.float32)  # Initialize with (x, y, vx, vy)
    
