#Importing all the modules
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

# checking for connection if there is not a secure one
# and then getting data from OpenML
if (not os.environ.get("PYTHONHTTPSVERIFY", "") and
getattr(ssl,'_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

# Fetching the data from dataset
X = np.load('image.npz')
X = X['arr_0']
y = pd.read_csv("labels.csv")
y = y["labels"]

# no of samples of each value
# print("Values of each class:")
# print(pd.Series(y).value_counts())

classes = ["A","B" ,"C" ,"D" ,"E" ,"F" ,"G" ,"H" ,"I" ,"J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
n_classes = len(classes)

#Splitting the data and scaling it
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=7500, test_size=2500)

#scaling the features to binary
scaled_X_train = X_train/255.0
scaled_X_test = X_test/255.0

#Fitting the training data into the model
clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(scaled_X_train, y_train)
# solver='saga' is used since large amount of data is present

#Calculating the accuracy of the model
y_predicted = clf.predict(scaled_X_test)
accuracy = accuracy_score(y_test, y_predicted)
print("The accuracy of the model is :- ",accuracy)

#Starting the camera
cap = cv2.VideoCapture(0)

while(True):
  # Capture frame-by-frame
  try:
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Drawing a box in the center of the video
    height, width = gray.shape
    upper_left = (int(width / 2 - 60), int(height / 2 - 60))
    bottom_right = (int(width / 2 + 60), int(height / 2 + 60))
    cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 2)

    #To only consider the area inside the box for detecting the alphabet
    roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]

    #Converting cv2 image to recognisable format
    im_pil = Image.fromarray(roi)

    # convert to grayscale image
    image_bw = im_pil.convert('L')
    image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)

    image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
    pixel_filter = 20
    #setting pixel value (min and max)
    min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0, 255)
    max_pixel = np.max(image_bw_resized_inverted)
    
    # converting to array and resizing it for model to recognise
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
    test_pred = clf.predict(test_sample)
    print("Predicted alphabet is: ", test_pred)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  except Exception as e:
    pass

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

