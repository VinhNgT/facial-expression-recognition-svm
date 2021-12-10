from sklearn import preprocessing
import emotion_clf
import cv2
import pickle
import numpy as np

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
with open(emotion_clf.MODEL_DATA_FILE, "rb") as file:
    model_data = pickle.load(file)
# with open(emotion_clf.SCALE_FILE, "rb") as file:
#     scaler = pickle.load(file)


def predict(landmark_vectorized):
    # Predict emotion
    npar_pd = np.array([landmark_vectorized])
    # npar_pd = scaler.transform(npar_pd)
    prediction_emo = model_data.predict(npar_pd)

    return prediction_emo[0]


def show_webcam_and_run():
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if cam.isOpened():
        ret, frame = cam.read()
    else:
        print("Không tìm thấy WebCam")
        return

    while ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe_image = clahe.apply(gray)
        faces_rect = emotion_clf.get_faces_rect(clahe_image)

        for rect in faces_rect:
            landmark = emotion_clf.get_landmark(rect, clahe_image)

            if landmark:
                landmark_vectorized = emotion_clf.vectorize_landmark(landmark)
                prediction = predict(landmark_vectorized)

                overlay_frame(frame, rect, emotion_clf.EMOTIONS_IN_DATASET[prediction])

                for point in landmark:
                    cv2.circle(
                        frame,
                        (point[0], point[1]),
                        1,
                        (100, 150, 200),
                        thickness=2,
                    )

        cv2.imshow("WEBCAM (Nhan phim 'Q' de thoat)", frame)
        ret, frame = cam.read()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


def overlay_frame(frame, face_rect, emotion):
    font_color = (0, 0, 0)
    if emotion == "angry":
        font_color = (0, 0, 255)
    if emotion == "happy":
        font_color = (0, 255, 0)
    if emotion == "sad":
        font_color = (255, 0, 0)

    cv2.rectangle(
        frame,
        (face_rect.left(), face_rect.top()),
        (face_rect.right(), face_rect.bottom()),
        font_color,
        thickness=4,
    )

    cv2.putText(
        frame,
        emotion,
        (face_rect.left(), face_rect.top() - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        font_color,
        2,
        cv2.LINE_AA,
    )

    return frame


if __name__ == "__main__":
    show_webcam_and_run()
