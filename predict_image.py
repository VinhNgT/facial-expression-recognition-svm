import run_clf as rclf
import cv2


def run_image(path):
    frame = cv2.imread(path)

    frame = rclf.predict_frame(frame)
    cv2.imshow("Anh", frame)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_image("D:\group.jpg")
