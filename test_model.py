import cv2
import dlib
import pickle
import numpy as np
import math

# Các bước nhận dạng:
# - Tách các khuôn mặt ra khỏi ảnh input, tìm đặc trưng khuôn mặt của chúng rồi vector hoá nó
# - Cho đặc trưng khuôn mặt vào MODEL_DATA_FILE nhận lại được kết quả là cảm xúc của khuôn mặt đó

MODEL_PREDICTOR_FILE = "model_data/shape_predictor_68_face_landmarks.dat"
MODEL_DATA_FILE = "model_data/model.pkl"
SHOW_FACE_DETAIL = False

# MODEL_PREDICTOR_FILE là bộ dữ liệu được huấn luyện sẵn để tách các đặc trưng mặt ra khỏi ảnh
# MODEL_DATA_FILE là bộ dữ liệu phân lớp đã được huấn luyện để phân biệt các cảm xúc

detector = dlib.get_frontal_face_detector()
model_predictor = dlib.shape_predictor(MODEL_PREDICTOR_FILE)
with open(MODEL_DATA_FILE, "rb") as file:
    model_data = pickle.load(file)


def get_faces_and_landmarks(image, frame, show_detail=SHOW_FACE_DETAIL):
    result = []

    # Tìm các khuôn mặt trong ảnh
    detections = detector(image, 1)

    # For all detected face instances individually
    # Tìm các đặc trưng khuôn mặt, vector hoá chúng
    for _, face in enumerate(detections):
        # Get facial landmarks with prediction model
        shape = model_predictor(image, face)
        xpoint = []
        ypoint = []
        for i in range(0, 68):
            # Chỉ hiện 2 điểm mũi
            # if (i == 27) | (i == 30):
            # For each point, draw a red circle with thickness2 on the original frame

            # Hiện hết
            if show_detail:
                cv2.circle(
                    frame,
                    (shape.part(i).x, shape.part(i).y),
                    1,
                    (0, 0, 255),
                    thickness=2,
                )

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

            # Get the angle the vector describes relative to the image, corrected for the offset that the nose-bridge
            # has when the face is not perfectly horizontal
            if x == xcenter:
                angle_relative = 0
            else:
                angle_relative = (
                    math.atan(float(y - ycenter) / (x - xcenter)) * 180 / math.pi
                ) - angle_nose

            landmarks.append(dist)
            landmarks.append(angle_relative)

        result.append((face, landmarks))

    # Trả lại các đặc trưng khuôn mặt và ảnh khuôn mặt kèm với chúng
    return result


def predict(landmarks, model):
    # Predict emotion
    npar_pd = np.array([landmarks])
    prediction_emo = model.predict(npar_pd)

    return prediction_emo[0]


def run_image(path, model):
    frame = cv2.imread(path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray)

    # Get Point and Landmarks
    faces_info = get_faces_and_landmarks(clahe_image, frame)

    if len(faces_info) == 0:
        pass
    else:
        for face, landmarks in faces_info:
            emotion = predict(landmarks, model)
            frame = overlay_frame(frame, face, emotion)

    cv2.imshow("Anh", frame)  # Display the frame

    while True:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


def show_webcam_and_run(model):
    # Set up some required objects
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Webcam objects

    if cam.isOpened():
        ret, frame = cam.read()
    else:
        print("Không tìm thấy WebCam")
        return

    while ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_image = clahe.apply(gray)

        # Get Point and Landmarks
        faces_info = get_faces_and_landmarks(clahe_image, frame)

        if len(faces_info) == 0:
            pass
        else:
            for face, landmarks in faces_info:
                emotion = predict(landmarks, model)
                frame = overlay_frame(frame, face, emotion)

        cv2.imshow("WEBCAM (Nhan phim 'Q' de thoat)", frame)  # Display the frame
        ret, frame = cam.read()

        if cv2.waitKey(1) & 0xFF == ord("q"):  # Exit program when user press 'q'
            break

    cam.release()
    cv2.destroyAllWindows()


def overlay_frame(frame, face, emotion):
    font_color = (0, 0, 0)
    if emotion == "angry":
        font_color = (0, 0, 255)
    if emotion == "happy":
        font_color = (0, 255, 0)
    if emotion == "sad":
        font_color = (255, 0, 0)

    cv2.rectangle(
        frame,
        (face.left(), face.top()),
        (face.right(), face.bottom()),
        font_color,
        thickness=4,
    )

    cv2.putText(
        frame,
        emotion,
        (face.left(), face.top() - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        font_color,
        2,
        cv2.LINE_AA,
    )

    return frame


if __name__ == "__main__":
    show_webcam_and_run(model_data)
    # run_image("D:/smile2.png", model_data)
