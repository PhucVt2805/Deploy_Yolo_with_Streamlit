import requests
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2
from io import BytesIO
from streamlit_webrtc import ClientSettings, VideoTransformerBase, WebRtcMode, webrtc_streamer

st.set_page_config(page_title="Thị giác máy tính", page_icon="🖥️")

# Khởi tạo mô hình YOLOv8
model = YOLO("yolov8n.pt")

# Định nghĩa hàm biến đổi video WebRTC
class ObjectDetector(VideoTransformerBase):
    def transform(self, frame):
        # Chuyển đổi khung hình thành hình ảnh
        img = cv2.cvtColor(frame.to_ndarray(), cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        results = model.predict(source=img)
        res_plotted = results[0].plot()
        output_frame = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        # Trả về khung ảnh sau khi đã được xử lí
        return res_plotted

# Định nghĩa ứng dụng Streamlit
def main():
    st.title("Ứng dụng nhận diện rác thải")
    # Tạo nút radio để chọn giữa tải hình ảnh lên, sử dụng webcam hoặc cung cấp URL hình ảnh
    choice = st.radio("Hãy chọn:", ("Tải lên 1 bức ảnh", "Sử dụng Webcam", "Cung cấp URL hình ảnh"))
    if choice == "Tải lên 1 bức ảnh":
        # Tạo widget tải lên 1 bức ảnh
        uploaded_file = st.file_uploader("Chọn một hình ảnh...", type=["jpg", "jpeg", "png"])
        # Nếu tệp tin đã được tải lên
        if uploaded_file is not None:
            # Tải hình ảnh từ tập tin đã tải lên
            img = Image.open(uploaded_file)
            results = model(source=img)
            res_plotted = results[0].plot()
            cv2.imwrite('image/test_image_output.jpg', res_plotted)
            col1, col2 = st.columns(2)
            col1.image(img, caption="Hình ảnh đã tải lên", use_column_width=True)
            # Hiển thị hình ảnh đã được tải lên
            col2.image('image/test_image_output.jpg', caption="Predected Image", use_column_width=True)

    elif choice == "Sử dụng Webcam":
        # Định nghĩa cài đặt máy khách WebRTC
        client_settings = ClientSettings(
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={
                "video": True,
                "audio": False,
            },
        )
        # Khởi động trình phát trực tuyến WebRTC
        webrtc_streamer(
            key="object-detection",
            mode=WebRtcMode.SENDRECV,
            client_settings=client_settings,
            video_transformer_factory=ObjectDetector,
        )

    elif choice == "Cung cấp URL hình ảnh":
        # Nhận URL hình ảnh từ người dùng
        image_url = st.text_input("Nhập URL hình ảnh:")

        # Nếu người dùng đã nhập URL hình ảnh
        if image_url != "":
            try:
                # Tải xuống hình ảnh từ URL
                response = requests.get(image_url)
                img = Image.open(BytesIO(response.content))

                results = model(source=img)
                res_plotted = results[0].plot()
                cv2.imwrite('image/test_image_output.jpg', res_plotted)

                col1, col2 = st.columns(2)
                col1.image(img, caption="Hình ảnh đã tải xuống" , use_column_width=True)
                # Hiển thị hình ảnh đã tải xuống
                col2.image('image/test_image_output.jpg', caption="Hình ảnh trả về sau khi nhận diện", use_column_width=True)
            except:
                st.error("Lỗi: URL hình ảnh không hợp lệ hoặc không thể tải hình ảnh xuống.")


if __name__ == '__main__':
    main()