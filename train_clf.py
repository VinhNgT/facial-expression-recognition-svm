"""
Module train_clf (Train Classifier)
Huấn luyện bộ phân lớp cảm xúc
"""

import expression_clf as eclf
import numpy as np
import pandas as pd
import pickle as pk

EMOTIONS_TO_TRAIN_FOR = ["angry", "happy", "sad", "surprise"]

# Load các dữ liệu trong bộ dữ liệu MODEL_DATASET và phân lớp chúng
# Chuẩn bị các dữ liệu cần thiết cho quá trình huấn luyện
def make_sets():
    df = pd.read_csv(eclf.DATASET)

    training_data = []
    training_label = []
    testing_data = []
    testing_label = []

    for _, row in df.iterrows():
        # Bỏ qua các dữ liệu không được thiết lập
        emotion_id = row["emotion"]
        if eclf.EMOTIONS_IN_DATASET[emotion_id] not in EMOTIONS_TO_TRAIN_FOR:
            continue

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
                    training_data.append(landmark_vectorized)
                    training_label.append(emotion_id)
                else:
                    testing_data.append(landmark_vectorized)
                    testing_label.append(emotion_id)

    # Ép kiểu sang dạng numpy
    training_data = np.array(training_data)
    training_label = np.array(training_label)
    testing_data = np.array(testing_data)
    testing_label = np.array(testing_label)

    return training_data, training_label, testing_data, testing_label


def train_model():
    print("Chuẩn bị dữ liệu cho quá trình huấn luyện...")
    X_train, y_train, X_test, y_test = make_sets()

    print("Huấn luyện...")
    eclf.clf.fit(X_train, y_train)

    print("Kiểm tra độ chính xác...")
    pred_accuracy = eclf.clf.score(X_test, y_test)
    test_pred = eclf.clf.predict(X_test)
    print("Độ chính xác = ", pred_accuracy * 100, "%")

    with open(eclf.CLF_FILE, "wb") as out:
        pk.dump(eclf.clf, out)

    print("Đã huấn luyện xong")


if __name__ == "__main__":
    train_model()
