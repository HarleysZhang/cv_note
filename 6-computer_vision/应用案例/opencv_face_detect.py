"""
Reference:
	1. https://github.com/CharlesPikachu/pydrawing
	2. https://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_face_detection.html#gsc.tab=0
	3. https://google.github.io/mediapipe/solutions/face_detection.html
    4. https://juejin.cn/post/7034325175021600782
"""

import cv2
import copy
from PIL import Image
import numpy as np

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# To capture video from webcam. 
cap = cv2.VideoCapture(0) # opencv 的函数，参数 0 代表调用电脑自带摄像，若改为 1 则调用外设摄像
hat_img_bgra = cv2.imread("./images/1024.png", -1) # 图像需为PNG格式（方便alpha通道的使用）
r, g, b, a = cv2.split(hat_img_bgra)
hat_rgb = cv2.merge((r, g, b)) # 把 rgb 三通道合成一张rgb的彩色图, shape is (height, width, channel)

def douyin_effect(image, x, y, w, h):
    face_roi = image[y: y+h, x: x+w] # create a ROI for the face
    face_roi_rgba = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGBA)
    
    # 通过把gb通道值置为0，生成r通道的图片
    image_arr_r = copy.deepcopy(face_roi_rgba)
    image_arr_r[:, :, 1:3] = 0
    image_r = Image.fromarray(image_arr_r).convert('RGBA')
    # 通过把r通道值置为0，生成gb通道的图片
    image_arr_gb = copy.deepcopy(face_roi_rgba)
    image_arr_gb[:, :, 0] = 0
    image_gb = Image.fromarray(image_arr_gb).convert('RGBA')
    
    # 生成一张黑色背景的画布图片，并把r通道图片复制粘贴在上面，粘贴的位置为(10,10)是为了与后面的gb通道图片错开
    canvas_r = Image.new('RGB', (face_roi.shape[1], face_roi.shape[0]), color=(0, 0, 0))
    canvas_r.paste(image_r, (15, 15), image_r)
    # gb通道图片的处理与上面类似
    canvas_gb = Image.new('RGB', (face_roi.shape[1], face_roi.shape[0]), color=(0, 0, 0))    
    canvas_gb.paste(image_gb, (0, 0), image_gb)
    
    add_image = np.array(canvas_gb) + np.array(canvas_r)
    output_image = cv2.cvtColor(add_image, cv2.COLOR_RGB2BGR)
    
    return output_image

def code_holiday_celebration(image, x, y, w, h, eyes_center):
    # 根据人脸大小调整节日 logo 大小(公式随意，比例一致即可)
    # ***********************************************************
    factor = 1 # 可调
    # ***********************************************************
    scaled_factor = w/hat_rgb.shape[1] # 缩放比例计算
    # 根据人脸大小缩放后的节日 logo 尺寸
    resized_hat_h = int(round(hat_rgb.shape[0] * scaled_factor * factor))
    resized_hat_w = int(round(hat_rgb.shape[1] * scaled_factor * factor))
    if resized_hat_h > y:
        resized_hat_h = y-1		#可调
    
    hat_resized = cv2.resize(hat_rgb, (resized_hat_w, resized_hat_h))
    
    mask = cv2.resize(a, (resized_hat_w, resized_hat_h))
    mask_inv = cv2.bitwise_not(mask)
 
    # 帽子相对于人脸框上线的偏移量
    # ***********************************************************
    dh = 0			# 可调
    dw = -60		# 可调
    # ***********************************************************
    # 原图ROI(这个公式原则上也可调)
    # print(eyes_center + dw - resized_hat_w//3)
    # print(eyes_center + dw + resized_hat_w//3*2)
    bg_roi = image[y + dh - resized_hat_h : y + dh, 
                  (eyes_center + dw - resized_hat_w//3):(eyes_center + dw + resized_hat_w//3*2)]
    # 原图ROI中提取放帽子的区域
    bg_roi = bg_roi.astype(float)
    mask_inv = cv2.merge((mask_inv, mask_inv, mask_inv))
    alpha = mask_inv.astype(float)/255
    # 相乘之前保证两者大小一致（可能会由于四舍五入原因不一致）
    alpha = cv2.resize(alpha, (bg_roi.shape[1], bg_roi.shape[0]))
    bg = cv2.multiply(alpha, bg_roi)
    bg = bg.astype('uint8')

    # 提取帽子区域
    hat = cv2.bitwise_and(hat_resized, hat_resized, mask=mask)
    # 添加圣诞帽
    hat = cv2.resize(hat, (bg_roi.shape[1], bg_roi.shape[0]))
    
    # 两个ROI区域相加
    add_hat = cv2.add(bg, hat)
    # 把添加好帽子的区域放回原图
    image[y + dh - resized_hat_h: y + dh, (eyes_center + dw - resized_hat_w//3):(eyes_center + dw + resized_hat_w//3*2)] = add_hat
    return image

def sketch_image(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.medianBlur(img_gray, 5)
    edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=5)
    ret, threshold = cv2.threshold(edges, 145, 255, cv2.THRESH_BINARY_INV)
    return threshold

def face_post_process(image, x, y, w, h, eyes_center):
    """Draw the rectangle and produce douyin effect around each face"""
    # 1, douyin effect
    douyin_face_roi = douyin_effect(image, x, y, w, h)
    image[y: y+h, x: x+w] = douyin_face_roi # douyin effect on image's face roi
    
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2) # visual face detect bbox
    # 2, 1024 programmer's Day Celebration
    code_holiday_image = code_holiday_celebration(image, x, y, w, h, eyes_center)
    # eyes = eye_cascade.detectMultiScale(gray[y: y+h, x: x+w])
    # 3, face carton effect
    # cartoonized_image = cartonize_image(code_holiday_image)
    
    return code_holiday_image

if __name__ == "__main__":
    while True:
        ret, image = cap.read() # Read the frame
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert to grayscale
        faces = face_cascade.detectMultiScale(gray, 1.1, 4) # Detect the faces
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y: y+h, x: x+w]
            eyes_center = int(x + w/2)
            post_process_image = face_post_process(image, x, y, w, h, eyes_center)  # face post-process and visual
            
            cv2.imshow('OpenCV Face Detection', post_process_image) # Display image
        
        # Stop if escape key is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    
    # Release the VideoCapture object
    cap.release()
    cv2.destroyAllWindows()