"""
Module expression_clf (Expression Classifier)
Chứa các biến và hàm cần thiết cho quá trình huấn luyện và nhận dạng cảm xúc
"""

import dlib
from cv2 import createCLAHE
from numpy.lib.function_base import angle
from sklearn.svm import SVC
import numpy as np
from math import cos, sin

# Các thông số cho bộ nhận dạng
LANDMARK_PREDICTOR_FILE = "model_data/shape_predictor_68_face_landmarks.dat"
DATASET = "model_data/fer2013.csv"
CLF_FILE = "model_data/clf.pkl"
EMOTIONS_IN_DATASET = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]

face_detector = dlib.get_frontal_face_detector()
model_predictor = dlib.shape_predictor(LANDMARK_PREDICTOR_FILE)
clahe = createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clf = SVC(C=10, cache_size=1000)


# Tìm các khuôn mặt trong ảnh, trả về khung (rect)
def get_faces_rect(image):
    detections = face_detector(image, 1)
    return detections


# Tìm đường bao các đặc điểm của MỘT khuôn mặt
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


# Vector hoá các đường bao, lấy điểm giữa mũi làm tâm (0, 0)
# Vector hoá đẩy khuôn mặt về tâm trục toạ độ xOy, mục đích giúp bộ phân lớp có thể phân biệt
# các khuôn mặt ở bất kỳ vị trí nào trong ảnh
def vectorize_landmark(landmark):
    tip_nose_id = 18

    landmark_vectorized = []
    for id, point in enumerate(landmark):
        if id != tip_nose_id:
            point_vector = (
                point[0] - landmark[tip_nose_id][0],
                point[1] - landmark[tip_nose_id][1],
            )
            landmark_vectorized.append(point_vector[0])
            landmark_vectorized.append(point_vector[1])

    return landmark_vectorized


# Căn chỉnh các vector để khử độ nghiêng của ảnh
# Giúp cân bằng khuôn mặt trong trường hợp trục x khuôn mặt không vuông góc với trục x camera
def align_landmark_vector(landmark_vetors):
    # Vector unit trục y
    unit_y_vector = (0, -1)

    top_nose = (landmark_vetors[15 * 2], landmark_vetors[15 * 2 + 1])

    # Vector mũi
    nose_vector = (
        top_nose[0] - 0,
        top_nose[1] - 0,
    )
    unit_nose_vector = nose_vector / np.linalg.norm(nose_vector)

    dot_product = np.dot(unit_y_vector, unit_nose_vector)

    # Góc mũi so với trục y của ảnh (radian)
    nose_angle = np.arccos(dot_product)

    # Quay tất cả vector để khử độ nghiêng của mặt
    theta = nose_angle
    rotation_matrix = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])

    result = []
    for i in range(0, len(landmark_vetors), 2):
        point = np.array((landmark_vetors[i], landmark_vetors[i + 1]))
        new_point = np.dot(rotation_matrix, point)

        result.append(new_point[0])
        result.append(new_point[1])

    return result


# Chuẩn hoá các vector thành khoảng -1 đến 1
# Chuẩn hoá vector giúp bộ phân lớp có thể phân biệt các khuôn mặt ở bất kỳ kích cỡ nào
def normalize_landmark_vector(landmark_vetors):
    result = []

    max_vector = abs(max(landmark_vetors, key=lambda vector: abs(vector)))
    for vector in landmark_vetors:
        new_vector = vector / max_vector
        result.append(new_vector)

    return result
