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
import matplotlib.pyplot as plt
import time

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# To capture video from webcam. 
cap = cv2.VideoCapture(0) # opencv 的函数，参数 0 代表调用电脑自带摄像，若改为 1 则调用外设摄像
hat_img_bgra = cv2.imread("./images/hat.png", -1) # 图像需为PNG格式（方便alpha通道的使用）
logo_img = cv2.imread('./images/10246.png') # shape is (862, 1264, 3)
# r, g, b, a = cv2.split(hat_img_bgra) # cv.split() 函数耗时很长，推荐用 numpy 多维数组索引
b, g, r, a = hat_img_bgra[:,:,0], hat_img_bgra[:,:,1], hat_img_bgra[:,:,2], hat_img_bgra[:,:,3]
hat_rgb = cv2.merge((r, g, b)) # 把 rgb 三通道合成一张rgb的彩色图, shape is (height, width, channel)
thresh_set = 45

####################################################################################################################
def runTime(func):
    """decorator: print the cost time of run function"""
    def wapper(arg, *args, **kwargs):
        start = time.time()
        res = func(arg, *args, **kwargs)
        end = time.time()
        print("="*80)
        print("function name: %s" %func.__name__)
        print("run time: %.4fs" %(end - start))
        print("="*80)
        return res
    return wapper

def plt_show_one(img1, title="fist image"):
    plt.figure('img1 roi',figsize=(25,25))
    plt.title(title)
    plt.imshow(cv2.cv2tColor(img1,cv2.COLOR_BGR2RGB))

def plt_show_two(img1, img2, title1="fist image", title2="second image"):
    plt.figure('img2 BINARY',figsize=(25,25))
    plt.subplot(121)
    plt.title(title1)
    plt.imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
    plt.subplot(122)
    plt.title(title2)
    plt.imshow(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB))

####################################################################################################################
@runTime
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

####################################################################################################################
@runTime
def add_logo(img1, x, y, w, h, img2):
    
    # 1, 将标志图像缩放到合适尺寸，并以此在原图上创建 ROI，同时将标志图像缩放，并显示
    # I want put img2 on img1's roi, So I create roi
    scaled_factor = w/img2.shape[1]
    resized_hat_h = int(round(img2.shape[0] * scaled_factor))
    img1_roi = img1[y - resized_hat_h : y, x : x + w] # 原ROI中提取放LOGO的区域, roi shape is (173, 253, 3)
    img2_resized = cv2.resize(img2, (img1_roi.shape[1], img1_roi.shape[0])) # 将img2缩放到roi一样大小
    print(img1_roi.shape, img2_resized.shape)

    # plt_show_two(img1_roi, img2_resized, "will be changed roi in img1", "resized img2")

    # 2，创建标志图像的mask和mask_inv图像, 并显示
    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
    # 80是img2背景的最大像素值
    ret, img2_mask = cv2.threshold(img2gray, thresh_set, 255,  cv2.THRESH_BINARY) # 像素灰度值大于80的取值255，反之取0
    ret, img2_mask_inv = cv2.threshold(img2gray, thresh_set, 255, cv2.THRESH_BINARY_INV) # 像素灰度值小于80的取值255，反之取0
    plt_show_two(img2_mask, img2_mask_inv, "img2 BINARY", "img2 BINARY_INV") # 显示图像

    # Masking（掩膜运算） 即图与掩膜的“按位与”运算: 原图中的每个像素和掩膜（Mask）中的每个对应像素进行按位与运算，
    # 如果为真，结果是原图的值【这是重点】；如果为假，结果就是零
    # mask的最大作用：让我们只关注我们感兴趣的图像部分。如下引用选自《Practical Python and OpenCV 3rd Edition》

    # 3，black-out the area of logo in ROI，and Take only region of logo from logo image, 并显示
    print(img2_mask_inv.shape, img1_roi.shape)
    img1_bg = cv2.bitwise_and(img1_roi, img1_roi, mask = img2_mask_inv) # img1_roi和img1_roi先按位与操作AND，结果还是img1_roi, 再将结果进行mask操作
    img2_fg = cv2.bitwise_and(img2_resized, img2_resized, mask = img2_mask) # 把前景以外的地方的像素置为0，即扣出LOGO
    # img2_fg[:,:,0] = 255
    # background 背景, foreground 前景
    plt_show_two(img1_bg, img2_fg, "img1 roi background", "img2 logo forground") # 显示图像

    # 4, Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)

    img1[y - resized_hat_h : y, x : x + w] = dst
    plt_show_two(dst, img1, "processed roi", "changed img1") # 显示图像
    cv2.imwrite('add_logo.png', img1)
    
    return img1

####################################################################################################################
@runTime
def add_hat(image, x, y, w, h, hat_rgb):
    center = int(x + w/2)
    # 根据人脸大小调整节日 LOGO 大小(公式随意，比例一致即可)
    # ***********************************************************
    factor = 1 # 可调, 越大 LOGO 也越大
    # ***********************************************************
    scaled_factor = w/hat_rgb.shape[1] # 缩放比例计算
    # 根据人脸大小缩放后的节日 LOGO 尺寸
    resized_hat_h = int(round(hat_rgb.shape[0] * scaled_factor * factor))
    resized_hat_w = int(round(hat_rgb.shape[1] * scaled_factor * factor))
    if resized_hat_h > y:
        resized_hat_h = y-1		#可调
    
    hat_resized = cv2.resize(hat_rgb, (resized_hat_w, resized_hat_h))
    
    mask = cv2.resize(a, (resized_hat_w, resized_hat_h))
    mask_inv = cv2.bitwise_not(mask)

    # 调整 LOGO 在头顶的位置，DW<0 则往左移，DW>0 则往右移
    # ***********************************************************
    dh = 0			# 可调
    dw = -60		# 可调
    # ***********************************************************
    # 原图ROI(这个公式原则上也可调)
    bg_roi = image[y + dh - resized_hat_h : y + dh, 
                  (center + dw - resized_hat_w//3):(center + dw + resized_hat_w//3*2)]
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
    print(bg.shape, hat.shape)
    # 两个ROI区域相加
    add_hat = cv2.add(bg, hat)
    # 把添加好帽子的区域放回原图
    image[y + dh - resized_hat_h: y + dh, (center + dw - resized_hat_w//3):(center + dw + resized_hat_w//3*2)] = add_hat
    return image

####################################################################################################################
@runTime
def face_post_process(img1, x, y, w, h):
    """Draw the rectangle and produce douyin effect around each face"""
    # 1, douyin effect
    douyin_face_roi = douyin_effect(img1, x, y, w, h)
    img1[y: y+h, x: x+w] = douyin_face_roi # douyin effect on image's face roi    
    # 2, 1024 programmer's Day Celebration
    out_img = add_logo(img1, x, y, w, h, logo_img)
    # out_img = add_hat(img1, x, y, w, h, hat_rgb)
    
    cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 0, 255), 2) # visual face detect bbox
    return out_img

####################################################################################################################
if __name__ == "__main__":
    while True:
        ret, image = cap.read() # Read the frame
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert to grayscale
        faces = face_cascade.detectMultiScale(gray, 1.1, 4) # Detect the faces
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y: y+h, x: x+w]

            post_process_image = face_post_process(image, x, y, w, h)  # face post-process and visual
            
            cv2.imshow('OpenCV Face Detection', post_process_image) # Display image
        
        # Stop if escape key is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    
    # Release the VideoCapture object
    cap.release()
    cv2.destroyAllWindows()