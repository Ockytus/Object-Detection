import cv2
import serial
from ultralytics import YOLO

# Cấu hình kết nối Serial
ser = serial.Serial('COM2', 9600)  # Thay 'COMx' bằng số cổng COM thích hợp và 9600 là tốc độ baud

model = YOLO(r'C:\Users\User\Desktop\Xu ly anh\NEW\runs\detect\train5\weights\best.pt')
vid = cv2.VideoCapture(0)

while (True):
    ret, frame = vid.read()
    frame = cv2.flip(frame, 1)  # mirror
    results = model(frame)
    annotated_image = results[0].plot()

    # Khởi tạo biến kiểm tra nhận diện các lớp
    detected_classes = {'Trong Nghia': False, 'Chai Nuoc': False, 'Duc Thang': False}

    for r in results:
        acls = r.boxes.cls
        for a in acls:
            cls_value = int(a.item())  # Lấy giá trị của lớp từ tensor
            if cls_value == 0:
                detected_classes['Trong Nghia'] = True
            elif cls_value == 1:
                detected_classes['Chai Nuoc'] = True
            elif cls_value == 2:
                detected_classes['Duc Thang'] = True

            # Gửi dữ liệu qua cổng COM
            ser.write(str(cls_value).encode())  # Chuyển số lớp thành chuỗi và gửi qua cổng COM

    # Kiểm tra các lớp đã được nhận diện và gửi mã tương ứng qua cổng COM
    for cls_name, detected in detected_classes.items():
        if not detected:
            if cls_name == 'Trong Nghia':
                ser.write(b'4')
            elif cls_name == 'Chai Nuoc':
                ser.write(b'5')
            elif cls_name == 'Duc Thang':
                ser.write(b'6')

    cv2.imshow("YOLOv8 Inference", annotated_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()