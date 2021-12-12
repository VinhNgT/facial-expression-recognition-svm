"""
Module run_clf (Run Classifier)
Chạy bộ phân lớp cảm xúc
"""

import expression_clf as eclf
import cv2
import pickle as pk
import numpy as np

# Mức độ hiện thông tin chi tiết khuôn mặt
# Lv 0: Không hiện
# Lv 1: Hiện đường bao đặc điểm khuôn mặt
# Lv 2: Hiện vector khuôn mặt
SHOW_FACE_DETAIL_LV = 2
DEFAULT_OVERLAY_COLOR = (0, 0, 255)  # red

# Load bộ phân lớp đã được huấn luyện
with open(eclf.CLF_FILE, "rb") as file:
    clf = pk.load(file)


# Nhận dạng bộ vector này là cảm xúc gì
def predict(vector):
    npar_pd = np.array([vector])
    prediction_emo = clf.predict(npar_pd)

    return prediction_emo[0]


def predict_frame(frame):
    # Chuyển ảnh sang dạng xám
    # Chạy clahe để cân bằng độ tương phản ảnh
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe_image = eclf.clahe.apply(gray)

    # Lấy các khuôn mặt trong ảnh
    detected_faces = eclf.get_faces_rect(clahe_image)

    for rect in detected_faces:
        # Lấy các đường bao đặc điểm khuôn mặt trong mỗi khuôn mặt nhận dạng được
        landmark = eclf.get_landmark(rect, clahe_image)

        # Nếu lấy được các đường bao...
        if landmark:
            # Vector hoá đường bao
            landmark_vectorized = eclf.vectorize_landmark(landmark)

            # Chuẩn hoá vector đường bao
            landmark_vectorized_normalized = eclf.normalize_landmark_vector(
                landmark_vectorized
            )

            # Chạy suy đoán
            prediction = predict(landmark_vectorized_normalized)

            if SHOW_FACE_DETAIL_LV == 1:
                overlay_landmark(frame, landmark)
            if SHOW_FACE_DETAIL_LV == 2:
                overlay_vector(frame, landmark_vectorized, landmark)

            # Vẽ kết quả
            overlay_prediction(frame, rect, eclf.EMOTIONS_IN_DATASET[prediction])

    return frame


# Vẽ vector lên ảnh
def overlay_vector(frame, vectors, landmark):
    start_point = landmark[18]

    for i in range(0, len(vectors), 2):
        end_point = (start_point[0] + vectors[i], start_point[1] + vectors[i + 1])

        cv2.arrowedLine(
            frame,
            start_point,
            end_point,
            DEFAULT_OVERLAY_COLOR,
            1,
        )


# Vẽ đường bao lên ảnh
def overlay_landmark(frame, landmark):
    for point in landmark:
        cv2.circle(
            frame,
            (point[0], point[1]),
            1,
            DEFAULT_OVERLAY_COLOR,
            thickness=2,
        )


# Vẽ kết quả nhận diện lên ảnh
def overlay_prediction(frame, face_rect, emotion):
    font_color = DEFAULT_OVERLAY_COLOR
    if emotion == "angry":
        font_color = (0, 0, 255)
    if emotion == "happy":
        font_color = (0, 255, 0)
    if emotion == "sad":
        font_color = (255, 0, 0)
    if emotion == "surprise":
        font_color = (255, 153, 255)

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
