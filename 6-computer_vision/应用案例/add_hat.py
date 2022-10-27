import cv2
import copy
from PIL import Image
import numpy as np

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
hat_img_bgra = cv2.imread("./images/hat.png", -1) # 图像需为PNG格式（方便alpha通道的使用）
r, g, b, a = cv2.split(hat_img_bgra)
hat_rgb = cv2.merge((r, g, b)) # 把 rgb 三通道合成一张rgb的彩色图, shape is (height, width, channel)

def add_hat(img1, x, y, w, h, hat_rgb):
    center = int(x + w/2)
    # I want to put logo on the head, So I create a ROI
    scaled_factor = w / hat_rgb.shape[1]
    resized_hat_h = int(round(hat_rgb.shape[0] * scaled_factor))
    bg_roi = img1[y - resized_hat_h : y, x : x + w]
    
    
    # 根据人脸大小调整节日 logo 大小(公式随意，比例一致即可)
    # ***********************************************************
    factor = 1.0 # 可调
    # ***********************************************************
    scaled_factor = w/hat_rgb.shape[1] # 缩放比例计算
    # 根据人脸大小缩放后的节日 logo 尺寸
    resized_hat_h = int(round(hat_rgb.shape[0] * scaled_factor))
    resized_hat_w = int(round(hat_rgb.shape[1] * scaled_factor * factor))
    if resized_hat_h > y:
        resized_hat_h = y-1		#可调
    
    hat_resized = cv2.resize(hat_rgb, (resized_hat_w, resized_hat_h))
    
    mask = cv2.resize(a, (resized_hat_w, resized_hat_h))
    mask_inv = cv2.bitwise_not(mask)
 
    # LOGO 相对于人脸框上线的偏移量
    # ***********************************************************
    dh = 0			# 可调
    dw = -10		# < 0，左移，> 0右移
    # ***********************************************************
    # 原图 ROI(这个公式原则上也可调)
    bg_roi = img1[y + dh - resized_hat_h : y + dh, 
                  (center + dw - resized_hat_w//3):(center + dw + resized_hat_w//3*2)]
    # 原图 ROI 中提取放 LOGO 的区域
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
    img1[y + dh - resized_hat_h: y + dh, (center + dw - resized_hat_w//3):(center + dw + resized_hat_w//3*2)] = add_hat
    return img1

if __name__ == "__main__":
    img = cv2.imread("./images/programmer.png") # 必须为 png 图片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.4, 5) # Detect the faces
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y: y+h, x: x+w]
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2) # visual face detect bbox
        output_image = add_hat(img, x, y, w, h, hat_rgb)
    
    cv2.imwrite('./images/add_hat.png', output_image)