"""
Module train_clf (Train Classifier)
Huấn luyện bộ phân lớp cảm xúc
"""

import expression_clf as eclf
import numpy as np
import pandas as pd
import pickle as pk
import os

# "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral",
EMOTIONS_TO_TRAIN_FOR = ["angry", "happy", "sad", "surprise"]
DATASET_CACHE = eclf.DATASET + "cache"


# Lớp lưu dataset đã qua xử lý
class DatasetCache:
    def __init__(self) -> None:
        self.training_data = []
        self.training_label = []
        self.testing_data = []
        self.testing_label = []

    def load(self, emotions_list):
        result_training_data = []
        result_training_label = []
        result_testing_data = []
        result_testing_label = []

        for data, label in zip(self.training_data, self.training_label):
            if eclf.EMOTIONS_IN_DATASET[label] in emotions_list:
                result_training_data.append(data)
                result_training_label.append(label)

        for data, label in zip(self.testing_data, self.testing_label):
            if eclf.EMOTIONS_IN_DATASET[label] in emotions_list:
                result_testing_data.append(data)
                result_testing_label.append(label)

        result_training_data = np.array(result_training_data)
        result_training_label = np.array(result_training_label)
        result_testing_data = np.array(result_testing_data)
        result_testing_label = np.array(result_testing_label)

        return (
            result_training_data,
            result_training_label,
            result_testing_data,
            result_testing_label,
        )


# Xử lý trước dataset, giúp việc load dataset nhanh hơn
# Load các dữ liệu trong bộ dữ liệu MODEL_DATASET và phân lớp chúng
# Chuẩn bị các dữ liệu cần thiết cho quá trình huấn luyện
def create_dataset_cache():
    dataset_cache = DatasetCache()
    df = pd.read_csv(eclf.DATASET)

    for _, row in df.iterrows():
        # Giải mã ảnh, các ảnh trong bộ dữ liệu là ảnh xám có kích thước 48x48
        # Chạy clahe để cân bằng độ tương phản ảnh
        image = np.fromstring(row["pixels"], dtype=np.uint8, sep=" ").reshape((48, 48))
        clahe_frame = eclf.clahe.apply(image)

        # Lấy các khuôn mặt trong ảnh
        detected_faces = eclf.get_faces_rect(clahe_frame)

        for face_rect in detected_faces:
            # Lấy các đường bao đặc điểm khuôn mặt trong mỗi khuôn mặt nhận dạng được
            landmark = eclf.get_landmark(face_rect, clahe_frame)

            # Nếu lấy được các đường bao...
            if landmark:
                # Vector hoá đường bao, căn chỉnh, chuẩn hoá vector
                landmark_vectorized = eclf.normalize_landmark_vector(
                    eclf.align_landmark_vector(eclf.vectorize_landmark(landmark))
                )

                # Nếu dữ liệu này là dữ liệu 'Training' thì cho vào bộ training, còn ko thì cho vào bộ test
                if row["Usage"] == "Training":
                    dataset_cache.training_data.append(landmark_vectorized)
                    dataset_cache.training_label.append(row["emotion"])
                else:
                    dataset_cache.testing_data.append(landmark_vectorized)
                    dataset_cache.testing_label.append(row["emotion"])

    with open(DATASET_CACHE, "wb") as out:
        pk.dump(dataset_cache, out)


def load_dataset_cache(emotions_list, force_renew_cache=False):
    # Tạo file cache cho dataset nếu chưa tồn tại
    if not os.path.isfile(DATASET_CACHE) or force_renew_cache:
        create_dataset_cache()

    with open(DATASET_CACHE, "rb") as file:
        dataset_cache = pk.load(file)
        return dataset_cache.load(emotions_list)


def train_model():
    print("Chuẩn bị dữ liệu cho quá trình huấn luyện...")
    X_train, y_train, X_test, y_test = load_dataset_cache(EMOTIONS_TO_TRAIN_FOR)

    print("Huấn luyện...")
    eclf.clf.fit(X_train, y_train)

    print("Kiểm tra độ chính xác...")
    pred_accuracy = eclf.clf.score(X_test, y_test)
    print("Độ chính xác = ", pred_accuracy * 100, "%")

    with open(eclf.CLF_FILE, "wb") as out:
        pk.dump(eclf.clf, out)

    print("Đã huấn luyện xong")


if __name__ == "__main__":
    train_model()
