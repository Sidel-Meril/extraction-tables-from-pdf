from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from matplotlib import pyplot as plt
import numpy as np, random
import cv2

image_path=r'D:\PyProjects\extremeportal\test_img/page0.jpg'
convert_to_cv = lambda img: np.stack((np.array(img),)*3, axis=-1)
convert_to_pil = lambda img: Image.fromarray(img)
img = Image.open(image_path)

def crop_image(img, left, top, right, bottom):
    resized_img = img.crop((left, top, right, bottom))
    return  resized_img

# _separation = 960
# width, height = img.size
# img = crop_image(img, 0, 0, _separation, height)

x, y, w, h = 224, 448, 480, 224
width, height = img.size
img = crop_image(img, x, y, x + w, y + h)

def clear_lines_and_boxes(frame, threshold_w, threshold_h):
    _h, _w, _c = frame.shape
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_img = (255 - gray_img)
    thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] # применяем цветовой фильтр
    # Remove horizontal
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if  w >= threshold_w:
            print('width', w)
            cv2.drawContours(frame, [c], -1, (255, 255, 255), -1)
            cv2.drawContours(frame, [c], -1, (255, 255, 255), 5)

    # Remove boxes
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if  h>threshold_h:
            print('box', w, h)
            cv2.drawContours(frame, [c], -1, (255, 255, 255), -1)
            cv2.drawContours(frame, [c], -1, (255, 255, 255), 5)

    # repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 6))
    # frame = 255 - cv2.morphologyEx(255 - frame, cv2.MORPH_CLOSE, repair_kernel, iterations=5)
    _resize_rate = int(_w / 600)
    _resize_rate = 1
    cv2.imshow('Deleted lines and boxes', cv2.resize(frame, (int(_w / _resize_rate), int(_h / _resize_rate))))

    return frame

def detect_fig(frame):

    obj = []
    count=0
    _h, _w, _c = frame.shape
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_img = (255 - gray_img)
    thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] # применяем цветовой фильтр
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    print(len(cnts))
    for c in cnts:
        cv2.drawContours(frame, [c], -1, (0, 0, 0), -1)


    # Draw rectangles, the 'area_treshold' value was determined empirically
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    print(len(cnts))
    area_treshold = (_h*_w)*0.0
    for c in cnts:
        if cv2.contourArea(c) > area_treshold:
            x, y, w, h = cv2.boundingRect(c)
            print(x, y, w, h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (36, 255, 12), 3)
            # cv2.rectangle(orig, (x, y), (x + w, y + h), (36, 255, 12), 3)
    _resize_rate = int(_h/600)

    return frame

def add_filters(test_im, resize = 2, blur = 5, rate=6):
    test_im = test_im.resize([int(s * 1 / resize) for s in test_im.size])
    test_im = test_im.filter(ImageFilter.BoxBlur(blur))
    test_im = test_im.resize([int(s * resize) for s in test_im.size])


    test_im = test_im.resize([int(s*1/(2**rate)) for s in test_im.size])
    for i in range(rate):
        filterr = ImageEnhance.Contrast(test_im)
        test_im = filterr.enhance(20)
        filterr = ImageEnhance.Sharpness(test_im)
        test_im = filterr.enhance(20)
        filterr = ImageEnhance.Contrast(test_im)
        test_im = filterr.enhance(20)
        test_im = test_im.resize([int(s*2) for s in test_im.size])
        test_im = ImageOps.grayscale(test_im)
    return test_im

def on_change_rate(val):
    imageCopy = img.copy()
    global resize_val, blur_val, rate_val
    rate_val = val
    imageCopy = convert_to_cv(add_filters(imageCopy, resize=resize_val, blur=blur_val, rate=rate_val))
    imageCopy = detect_fig(imageCopy)
    cv2.imshow(windowName, cv2.resize(imageCopy, (int(_w/_resize_rate), int(_h/_resize_rate))))

def on_change_resize(val):
    imageCopy = img.copy()
    global resize_val, blur_val, rate_val
    resize_val = val
    imageCopy = convert_to_cv(add_filters(imageCopy, resize=resize_val, blur=blur_val, rate=rate_val))
    imageCopy = detect_fig(imageCopy)
    cv2.imshow(windowName, cv2.resize(imageCopy, (int(_w/_resize_rate), int(_h/_resize_rate))))

def on_change_blur(val):
    imageCopy = img.copy()
    global resize_val, blur_val, rate_val
    blur_val = val
    imageCopy = convert_to_cv(add_filters(imageCopy, resize=resize_val, blur=blur_val, rate=rate_val))
    imageCopy = detect_fig(imageCopy)
    cv2.imshow(windowName, cv2.resize(imageCopy, (int(_w/_resize_rate), int(_h/_resize_rate))))

_img = np.array(img)
windowName = 'image'
print(_img.shape)
_h, _w, _c = _img.shape
_resize_rate = int(_w / 200)
_resize_rate = 1

cv2.imshow('Original', cv2.resize(_img, (int(_w / _resize_rate), int(_h / _resize_rate))))
_img = clear_lines_and_boxes(_img, 500, 40)
img=convert_to_pil(_img)
resize_val, blur_val, rate_val = 1, 0, 0

cv2.imshow(windowName, cv2.resize(_img, (int(_w/_resize_rate), int(_h/_resize_rate))))
cv2.createTrackbar('resize', windowName, 1, 100, on_change_resize)
cv2.createTrackbar('blur', windowName, 1, 100, on_change_blur)
cv2.createTrackbar('rate', windowName, 1, 100, on_change_rate)

cv2.waitKey(0)
cv2.destroyAllWindows()