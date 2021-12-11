import dlib
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pickle
import pandas as pd

# Config section
MODEL_PREDICTOR_FILE = "model_data/shape_predictor_68_face_landmarks.dat"
MODEL_DATA_FILE = "model_data/model.pkl"
SCALE_FILE = "model_data/scale.pkl"
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
EMOTIONS_TO_TRAIN_FOR = ["angry", "happy", "sad", "surprise"]

detector = dlib.get_frontal_face_detector()
model_predictor = dlib.shape_predictor(MODEL_PREDICTOR_FILE)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clf = make_pipeline(
    StandardScaler(),
    SVC(kernel="linear", decision_function_shape="ovo", cache_size=1000),
)


def get_faces_rect(image):
    detections = detector(image, 1)
    return detections


def get_landmark(face_rect, image):
    landmarks_full = model_predictor(image, face_rect)
    landmarks_selected = []

    for point_id in range(landmarks_full.num_parts):
        # Bỏ qua đường biên xung quanh mặt do nó không biểu thị cảm xúc, bớt lãng phí tài nguyên
        # Chỉ lấy các đường biên: Mắt, Lông mày, Mũi, Miệng, Cằm
        if point_id <= 16 and point_id not in range(6, 11):
            continue

        landmarks_selected.append(
            (landmarks_full.part(point_id).x, landmarks_full.part(point_id).y)
        )

    return landmarks_selected


def vectorize_landmark(landmark):
    tip_nose = landmark[13]
    landmark_vectorized = []
    for id, point in enumerate(landmark):
        if id != 13:
            point_vector = (point[0] - tip_nose[0], point[1] - tip_nose[1])
            landmark_vectorized.append(point_vector[0])
            landmark_vectorized.append(point_vector[1])

    return landmark_vectorized


def __make_sets__():
    df = pd.read_csv(MODEL_DATASET)

    training_data = []
    training_label = []
    testing_data = []
    testing_label = []

    for _, row in df.iterrows():
        emotion_id = row["emotion"]
        if EMOTIONS_IN_DATASET[emotion_id] not in EMOTIONS_TO_TRAIN_FOR:
            continue

        image = np.fromstring(row["pixels"], dtype=np.uint8, sep=" ").reshape((48, 48))
        clahe_frame = clahe.apply(image)
        detected_faces = get_faces_rect(clahe_frame)

        for face_rect in detected_faces:
            landmark = get_landmark(face_rect, clahe_frame)

            if landmark:
                landmark_vectorized = vectorize_landmark(landmark)

                if row["Usage"] == "Training":
                    training_data.append(landmark_vectorized)
                    training_label.append(emotion_id)
                else:
                    testing_data.append(landmark_vectorized)
                    testing_label.append(emotion_id)

    training_data = np.array(training_data)
    training_label = np.array(training_label)
    testing_data = np.array(testing_data)
    testing_label = np.array(testing_label)

    return training_data, training_label, testing_data, testing_label


def train_model():
    print("Chuẩn bị bộ huấn luyện...")
    X_train, y_train, X_test, y_test = __make_sets__()

    print("Huấn luyện...")
    clf.fit(X_train, y_train)

    print("Kiểm tra độ chính xác...")
    pred_accuracy = clf.score(X_test, y_test)
    test_pred = clf.predict(X_test)
    print(confusion_matrix(y_test, test_pred))
    print(classification_report(y_test, test_pred))
    print("Độ chính xác = ", pred_accuracy * 100, "%")
    print("Đã huấn luyện xong")

    with open(MODEL_DATA_FILE, "wb") as out:
        pickle.dump(clf, out)


def debug():
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if cam.isOpened():
        ret, frame = cam.read()
    else:
        print("Không tìm thấy WebCam")
        return

    while ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe_image = clahe.apply(gray)
        faces_rect = get_faces_rect(clahe_image)

        for rect in faces_rect:
            landmark = get_landmark(rect, clahe_image)

            for id, point in enumerate(landmark):
                color = (0, 0, 255)
                # if id in range(6, 11):
                #     color = (0, 255, 0)

                cv2.circle(
                    frame,
                    (point[0], point[1]),
                    1,
                    color,
                    thickness=2,
                )

        cv2.imshow("WEBCAM (Nhan phim 'Q' de thoat)", frame)
        ret, frame = cam.read()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()
    pass


if __name__ == "__main__":
    train_model()
    # debug()
