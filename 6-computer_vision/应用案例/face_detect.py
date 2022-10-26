import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if __name__ == "__main__":
    img = cv2.imread("./images/student.png") # 必须为 png 图片

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    # 第二个参数值越大能检测出的人脸越多，但是有可能误检
    faces = face_cascade.detectMultiScale(gray, 1.2, 4) # Detect the faces
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2) # drae red face detect bbox
    cv2.imwrite('./images/face_detect.png', img)