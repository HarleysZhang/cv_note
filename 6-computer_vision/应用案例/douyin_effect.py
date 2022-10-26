import cv2
import copy
from PIL import Image
import numpy as np

def douyin_effect(img):
    # face= image[y: y+h, x: x+w] # create a ROI for the face
    img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    
    # 通过把gb通道值置为0，生成r通道的图片
    image_arr_r = copy.deepcopy(img_rgba)
    image_arr_r[:, :, 1:3] = 0
    image_r = Image.fromarray(image_arr_r).convert('RGBA')
    # 通过把r通道值置为0，生成gb通道的图片
    image_arr_gb = copy.deepcopy(img_rgba)
    image_arr_gb[:, :, 0] = 0
    image_gb = Image.fromarray(image_arr_gb).convert('RGBA')
    
    # 生成一张黑色背景的画布图片，并把r通道图片复制粘贴在上面，粘贴的位置为(10,10)是为了与后面的gb通道图片错开
    # 第二个参数值越大，错位效果越明显越晃眼睛
    canvas_r = Image.new('RGB', (img.shape[1], img.shape[0]), color=(0, 0, 0))
    canvas_r.paste(image_r, (10, 10), image_r)
    # gb通道图片的处理与上面类似
    canvas_gb = Image.new('RGB', (img.shape[1], img.shape[0]), color=(0, 0, 0))    
    canvas_gb.paste(image_gb, (0, 0), image_gb)
    
    add_image = np.array(canvas_gb) + np.array(canvas_r)
    output_image = cv2.cvtColor(add_image, cv2.COLOR_RGB2BGR)
    
    return output_image

if __name__ == "__main__":
    img = cv2.imread("./images/dog.png") # 必须为 png 图片
    output_image = douyin_effect(img)
    cv2.imwrite('./images/douyin_effect.png', output_image)