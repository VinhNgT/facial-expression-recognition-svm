import os
import dlib
import cv2
import numpy as np
import math
from sklearn.svm import SVC
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

MODEL_PREDICTOR_FILE = "model_data/shape_predictor_68_face_landmarks.dat"
MODEL_DATA_FILE = "model_data/model.pkl"
MODEL_DATASET = "model_data/fer2013.csv"
EMOTIONS_IN_DATASET = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]
EMOTIONS_TO_TRAIN_FOR = ["angry", "happy", "sad"]

detector = dlib.get_frontal_face_detector()
model = dlib.shape_predictor(MODEL_PREDICTOR_FILE)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Set the classifier as a support vector machines with linear kernel
clf = SVC(C=0.01, kernel="linear", decision_function_shape="ovo", probability=True)


def get_landmarks(image):
    detections = detector(image, 1)
    # For all detected face instances individually
    for k, d in enumerate(detections):
        # Get facial landmarks with prediction model
        shape = model(image, d)
        xpoint = []
        ypoint = []
        for i in range(0, 68):
            xpoint.append(float(shape.part(i).x))
            ypoint.append(float(shape.part(i).y))

        # Center points of both axis
        xcenter = np.mean(xpoint)
        ycenter = np.mean(ypoint)
        # Calculate distance between particular points and center point
        xdistcent = [(x - xcenter) for x in xpoint]
        ydistcent = [(y - ycenter) for y in ypoint]

        # Prevent divided by 0 value
        if xpoint[11] == xpoint[14]:
            angle_nose = 0
        else:
            # Point 14 is the tip of the nose, point 11 is the top of the nose brigde
            angle_nose = int(
                math.atan((ypoint[11] - ypoint[14]) / (xpoint[11] - xpoint[14]))
                * 180
                / math.pi
            )

        # Get offset by finding how the nose brigde should be rotated to become perpendicular to the horizontal plane
        if angle_nose < 0:
            angle_nose += 90
        else:
            angle_nose -= 90

        landmarks = []
        for cx, cy, x, y in zip(xdistcent, ydistcent, xpoint, ypoint):
            # Add the coordinates relative to the centre of gravity
            landmarks.append(cx)
            landmarks.append(cy)

            # Get the euclidean distance between each point and the centre point (the vector length)
            meanar = np.asarray((ycenter, xcenter))
            centpar = np.asarray((y, x))
            dist = np.linalg.norm(centpar - meanar)

            # Get the angle the vector describes relative to the image, corrected for the offset that the nosebrigde
            # has when the face is not perfectly horizontal
            if x == xcenter:
                angle_relative = 0
            else:
                angle_relative = (
                    math.atan(float(y - ycenter) / (x - xcenter)) * 180 / math.pi
                ) - angle_nose
            landmarks.append(dist)
            landmarks.append(angle_relative)

    if len(detections) < 1:
        # In case no case selected, print "error" values
        landmarks = "error"
    return landmarks


def make_sets():
    training_data = []
    training_label = []
    testing_data = []
    testing_label = []

    data = pd.read_csv(MODEL_DATASET)
    pixels = []

    for pixel in data["pixels"]:
        pixels.append(np.fromstring(pixel, dtype=np.uint8, sep=" ").reshape((48, 48)))

    for index, value in enumerate(pixels):
        if EMOTIONS_IN_DATASET[data["emotion"][index]] in EMOTIONS_TO_TRAIN_FOR:
            clahe_img = clahe.apply(value)
            landmarks_vec = get_landmarks(clahe_img)

            if landmarks_vec == "error":
                pass
            else:
                if data["Usage"][index] == "Training":
                    training_data.append(landmarks_vec)
                    training_label.append(EMOTIONS_IN_DATASET[data["emotion"][index]])
                else:
                    testing_data.append(landmarks_vec)
                    testing_label.append(EMOTIONS_IN_DATASET[data["emotion"][index]])
    return training_data, training_label, testing_data, testing_label


def create_model():
    print("Chuẩn bị bộ huấn luyện...")
    X_train, y_train, X_test, y_test = make_sets()

    # Turn the training set into a numpy array for the classifier
    np_X_train = np.array(X_train)
    np_y_train = np.array(y_train)

    # Train SVM
    print("Huấn luyện...")
    clf.fit(np_X_train, np_y_train)

    np_X_test = np.array(X_test)
    np_y_test = np.array(y_test)
    # Use score() function to get accuracy
    print("Kiểm tra độ chính xác...")
    pred_accuracy = clf.score(np_X_test, np_y_test)
    test_pred = clf.predict(np_X_test)

    print(confusion_matrix(np_y_test, test_pred))
    print(classification_report(np_y_test, test_pred))
    print("Độ chính xác = ", pred_accuracy * 100, "%")
    print("Đã huấn luyện xong")


if __name__ == "__main__":
    create_model()
    model_file = MODEL_DATA_FILE
    try:
        os.remove(model_file)
    except OSError:
        pass
    output = open(model_file, "wb")
    pickle.dump(clf, output)
    output.close()
