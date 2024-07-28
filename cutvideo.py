import cv2
def FrameCapture(Path):
    vidObj = cv2.VideoCapture(Path)
    count = 0
    success = 1
    while success:
        success, image = vidObj.read()
        if count % 10 == 0:
            cv2.imwrite("dataset\\frame%d.jpg"%(count/15),image)
        count+=1
print("0")
#if __name__ == '__name__':
FrameCapture(r"C:\Users\User\Desktop\Xu ly anh\NEW\WIN_20240321_13_41_09_Pro.mp4")
print("1")