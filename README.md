# Nhận diện biểu cảm khuôn mặt sử dụng SVM

## Hướng dẫn cài đặt môi trường

1. Cài đặt [Python 3.7.9](https://www.python.org/downloads/release/python-379/) và [Visual Studio C++](https://visualstudio.microsoft.com/vs/features/cplusplus/)

2. Tải toàn bộ code về bằng `git clone` hoặc Code -> Download ZIP rồi giải nén

3. Vào folder `facial-expression-recognition-svm` vừa có được

4. Mở Terminal:\
    Windows 11: Chuột phải -> Open in Windows Terminal\
    Windows 10: Shift + Chuột phải -> Open PowerShell window here

5. Cho phép Terminal chạy script:
    ```
    Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy remotesigned
    ```

6. Tạo môi trường python ảo, rồi kích hoạt nó:
    ```
    python -m venv .venv
    ./.venv/Scripts/Activate.ps1
    ```

7. Cài đặt các dependency cần thiết:
    ```
    pip install -r requirements.txt
    ```

- Nếu không báo lỗi, chương trình đã sẵn sàng được sử dụng
- Trong trường hợp chúng ta muốn khôi phục lại trạng thái gốc của Terminal:

    ```
    Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy Undefined
    deactivate
    ```

## Hướng dẫn sử dụng

Trước khi sử dụng, cần download bộ nhận dạng và dataset [tại đây](https://www.mediafire.com/file/9c9rzh7wxu6h11u/model_data.rar/file)\
Giải nén `model_data` rồi cho vào folder `facial-expression-recognition-svm`

### Huấn luyện
Chương trình có thể nhận diện được 7 loại cảm xúc khác nhau:\
**angry:angry:,
disgust:vomiting_face:,
fear:fearful:,
happy:smile:,
sad:cry:,
surprise:open_mouth:,
neutral:neutral_face:**

Tuy nhiên, càng nhiều cảm xúc mà chương trình có thể nhận dạng thì độ chính xác sẽ càng thấp, cho nên chúng ta sẽ chỉ huấn luyện nhận dạng một lượng cảm xúc nhất định thôi để tăng độ chính xác chung cho toàn bộ chương trình:

- Mở file `train_model.py`
- Ở phần `EMOTIONS_TO_TRAIN_FOR`, thiết lập các loại cảm xúc mà chương trình sẽ nhận dạng, mặc định là `["angry", "happy", "sad"]`
- Chọn trong danh sách các cảm xúc sau: `angry, disgust, fear, happy, sad, surprise, neutral`
- Bắt đầu quá trình huấn luyện:

    ```
    python train_model.py
    ```
- Nếu chọn càng nhiều cảm xúc để huấn luyện thì thời gian huấn luyện sẽ càng lâu

### Chạy nhận dạng
- Chạy nhận dạng sử dụng camera:

    ```
    python run_model.py
    ```
- Chạy nhận dạng từ file ảnh:
    - Mở file `run_model.py`
    - Ở cuối file, comment out hoặc xoá `show_webcam_and_run()`
    - Thiết lập `run_image("<Đường dẫn tới ảnh>")`
    - Chạy:
    
        ```
        python run_model.py
        ```

- Để hiện thị chi tiết các đặc trưng khuôn mặt, đặt `SHOW_FACE_DETAIL = True`

## Credits
- https://github.com/tupm2208/facial-emotion-recognition-svm
- https://github.com/amineHorseman/facial-expression-recognition-svm

    *Nguyễn Thế Vinh - CNT59ĐH - Đại học Hàng Hải Việt Nam*
