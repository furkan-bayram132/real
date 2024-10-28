#https://pysource.com/2021/10/29/kalman-filter-predict-the-trajectory-of-an-object/
import cv2
import numpy as np


class KalmanFilter:
    #4 state variables , 2 measurement variables
    kf = cv2.KalmanFilter(4, 2)
    #2x4 measurement matrix x ve y degerlerini 4 boyutlu state vectore donusturuyor (2 row var
    #cunku measurement ile sadece x ve y degelerlerini olcebiliyoruz hizi degil hizi hesaplicaz
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
    # bu 1.row x konumu + x hizi ile xk+1. pozisyonu hesapliyor
    # bu 2.row y konumu + y hizi ile yk+1. pozisyonu hesapliyor
    #state matriximizin 3. ve 4. ogeleri hiz oldugundan onlar degismeden kaliyor? hiz nasil degisecek.
    #
    kf.transitionMatrix = np.array([[1, 0, 1, 0], 
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)


    def predict(self, coordX, coordY):
            ''' This function estimates the position of the object'''
            measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
            self.kf.correct(measured)
            predicted = self.kf.predict()

            x, y = int(predicted[0, 0]), int(predicted[1, 0])
            return x, y

