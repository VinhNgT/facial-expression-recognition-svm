import run_clf as rclf
import cv2


def run_video(path):
    vid = cv2.VideoCapture(path)

    if not vid.isOpened():
        print("Không mở được Video !")
        return

    while True:
        ret, frame = vid.read()
        if not ret:
            return

        frame = rclf.predict_frame(frame)
        cv2.imshow("Video (Nhan phim 'Q' de thoat)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vid.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_video("data/linus.mp4")
