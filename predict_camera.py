import run_clf as rclf
import cv2


def run_camera():
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cam.isOpened():
        print("Không tìm thấy Webcam !")
        return

    while True:
        ret, frame = cam.read()
        if not ret:
            return

        frame = rclf.predict_frame(frame)
        cv2.imshow("Webcam (Nhan phim 'Q' de thoat)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera()
