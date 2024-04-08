import cv2

# 读取彩色图片
image = cv2.imread('100080.jpg')

# 将彩色图片转换为灰度图片
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 保存灰度图片
cv2.imwrite('2.png', gray_image)