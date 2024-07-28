import cv2
from ultralytics import YOLO

model = YOLO(r'C:\Users\User\Desktop\Xu ly anh\NEW\runs\detect\train5\weights\best.pt')
vid = cv2.VideoCapture(0)
#vid = cv2.VideoCapture(r'C:\Users\User\Desktop\Xu ly anh\NEW\WIN_20240321_13_41_09_Pro.mp4')

while (True):
    ret, frame = vid.read()
    frame = cv2.flip(frame, 1) #mirror
    results = model(frame)
    annotated_image = results[0].plot() #
   
    cv2.imshow("YOLOv8 Inference", annotated_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()