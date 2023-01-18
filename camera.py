import cv2
from model import DigitModel
import numpy as np


model = DigitModel('svc_model')


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(3, 640)
        self.video.set(4, 480)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        # Initialize the webcam


        while True:
            ret, frame = self.video.read()
            if not ret:
                continue
            # Convert the frame to grayscale

            grey = cv2.cvtColor(frame.copy(), cv2.COLOR_BGRA2GRAY)
            ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            preprocessed_digits = []
            for i,c in enumerate(contours):
                x, y, w, h = cv2.boundingRect(c)

                # Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

                # Cropping out the digit from the image corresponding to the current contours in the for loop
                digit = thresh[y:y + h, x:x + w]

                # Resizing that digit to (18, 18)
                resized_digit = cv2.resize(digit, (18, 18))

                # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
                padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)

                # Adding the preprocessed digit to the list of preprocessed digits
                preprocessed_digits.append(padded_digit)

                label =str(model.predict_digit(preprocessed_digits[i].reshape(1, 28*28))[0])

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Show the frame with the bounding box and prediction label in a window
            _, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()










