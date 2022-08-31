"""
Script based on OpenCV & Pillow library,
which implement extraction of table columns and rows which are not separated with lines.
"""

from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from matplotlib import pyplot as plt
import numpy as np
import cv2, os
from itertools import combinations

poppler_path = r'./poppler-0.68.0_x86/poppler-0.68.0/bin'

def read_img(image_path):
    """
    Reads image from declared path
    :return: PIL image, CV2 image
    """
    img = Image.open(image_path) #pil image
    _img = np.array(img) #cv image

    return img, _img


def convert_pdf2png(file= 'file.pdf', start=68, end=68):
    from pdf2image import convert_from_path
    try:
        os.mkdir('tmp')
        print('Folder ./tmp created.')
    except FileExistsError:
        print('Folder ./tmp already exist.')
    # Store Pdf with convert_from_path function
    images = convert_from_path(file,
                               first_page=start,
                               last_page=end,
                               poppler_path=poppler_path)

    for i,j in zip(range(len(images)),range(start, end+1)):
        # Save pages as images in the pdf
        images[i].save('./tmp/'+'page' + str(j) + '.jpg', 'JPEG')

# Helper functions
# OpenCV and Pillow work with images in different way.
# Pillow lib is used for applying filter to image like Contrast and Sharpness
# OpenCV is used for defining blocks of text/columns
convert_to_cv = lambda img: np.stack((np.array(img),)*3, axis=-1)
convert_to_pil = lambda img: Image.fromarray(img)
def im_show(images: list, ncol: int = 4, _titles: list = None) -> None:
    """
    Helper function for displaying set of images Pillow type
    """
    ncol, nrows = ncol, len(images)//ncol
    fig, ax = plt.subplots(nrows, ncol)
    fig.suptitle('Image')
    try:
        ax = ax.flatten()
        for i in range(ax.shape[0]):
            ax[i].imshow(np.array(images[i]), cmap='Greys', interpolation='nearest')
            if _titles != None: ax[i].set_xlabel(_titles[i])
            else: ax[i].axis("off")
    except AttributeError:
        for i in range(len(images)):
            ax.axis("off")
            ax.imshow(np.array(images[i]), cmap='Greys', interpolation='nearest')
            if _titles != None: ax.set_xlabel(_titles[i])
            else:
                ax.axis("off")

    plt.show()

def crop_image(img, left, top, right, bottom):
    """
    Cropping images Pillow type
    :param img: Pillow image
    :return: cropped Pillow image
    """
    resized_img = img.crop((left, top, right, bottom))
    return  resized_img

def merging(boxes):
    """
    Funtion which merges cells to rows
    :param boxes: list of boxes coords like (x,y,w,h)
    :return: list of merged boxes
    """
    # print('Start merging')
    _boxes = boxes.copy()
    for box1 in boxes:
        for box2 in boxes:
            if box1 == box2: continue
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2
            box = [box1, box2]
            x, y, h, w = [x1, x2], [y1, y2], [h1, h1], [w1, w2]
            xmin, xmax = np.argmin(x), np.argmax(x)
            ymin, ymax = np.argmin(y), np.argmax(y)
            if (y[ymin] + h[ymin] - y[ymax]) > 15:
                for trash in [box1, box2]:
                    try:
                        _boxes.remove(trash)
                        # print('Value deleted, now in', len(_boxes))
                    except ValueError:
                        # print('Value not exist')
                        pass
                _boxes.append(
                    [min(x), min(y), max([x[xmax] + w[xmax], x[xmin] + w[xmin]]) - min(x),
                     max([y[ymax] + h[ymax], y[ymin] + h[ymin]]) - min(y)])
                # print('Value added, now in', len(_boxes))
                return _boxes
    return _boxes

def merge_in_rows(boxes):
    """
    Merging boxes coordinates in rows with helper function
    until unmerged cell is present
    """
    if len(boxes)==1:
        return boxes
    merged_present = False
    old_boxes = []
    new_boxes = boxes.copy()
    while abs(len(old_boxes)-len(new_boxes))>0:
        # print('Start of iteration')
        old_boxes = new_boxes.copy()
        new_boxes = merging(old_boxes)
        # print('Res', len(old_boxes), len(new_boxes))


    # print('Merged:', len(old_boxes)-len(new_boxes))
    return new_boxes
def overlap(box1, box2):
    """
    Checking of boxes' overlap
    :return: True if boxes overlap, True
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    box = [box1, box2]
    x, y, h, w = [x1,x2], [y1, y2], [h1,h1], [w1,w2]
    interception = False
    inner = None
    merged = None
    _inner= []
    xmin, xmax = np.argmin(x), np.argmax(x)
    ymin, ymax = np.argmin(y), np.argmax(y)
    if (x[xmin]+w[xmin] - x[xmax])>10 and (y[ymin]+h[ymin] - y[ymax])>10:
        print(x[xmin]+w[xmin] - x[xmax])
        interception = True

    print(interception, inner)
    return interception, inner

def group_boxes_to_axis(boxes, axis, margin):
    """
    Join boxes by margin in defined axis
    :param boxes: list of boxes' coordinates
    :param axis: 0 - y, 1 - x
    :return: list of boxes' groups
    """
    groups={box:set() for box in boxes}
    for box1 in boxes:
        for box2 in boxes:
            if box1 == box2: continue
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2
            box = [box1, box2]
            x, y, h, w = [x1, x2], [y1, y2], [h1, h1], [w1, w2]
            xmin, xmax = np.argmin(x), np.argmax(x)
            ymin, ymax = np.argmin(y), np.argmax(y)
            if axis == 1:
                condition = y[ymin] + h[ymin] - y[ymax]
            elif axis == 0:
                condition = x[xmin] + w[xmin] - x[xmax]
            if condition > margin:
                groups[box1].add(box2)

    groups_list = []
    for key, values in groups.items():
        _values = values
        _values.add(key)
        groups_list.append(tuple(_values))
    return list(set(groups_list))

def crop_image_to_boxes(img, boxes, y_margin = 0, x_margin = 0):
    """
    Crop PIL image to boxes with list of boxes coords
    :return: list of box images
    """
    boxes_img = []
    for box in boxes:
        x, y, w, h = box
        width, height = img.size
        box_img = crop_image(img, x - x_margin, y - y_margin, x + w + x_margin, y + h + y_margin)
        boxes_img.append(box_img)
    return boxes_img

#image clearing
def clear_lines_and_boxes(frame, threshold_w, threshold_h, _max_window_size=600):
    """
    Clearing image from horizontal lines and black boxes.
    :param frame: image OpenCV type
    :param threshold_w: minimum value of line width which are deleting
    :param threshold_h: minimum value of box hight which are deleting
    :return: cleared OpenCV image
    """
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
            cv2.drawContours(frame, [c], -1, (255, 255, 255), -1)
            cv2.drawContours(frame, [c], -1, (255, 255, 255), 5)

    # Remove boxes
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if  h>threshold_h:
            cv2.drawContours(frame, [c], -1, (255, 255, 255), -1)
            cv2.drawContours(frame, [c], -1, (255, 255, 255), 5)

    # Repairing image
    # repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 6))
    # frame = 255 - cv2.morphologyEx(255 - frame, cv2.MORPH_CLOSE, repair_kernel, iterations=5)

    if _max_window_size!= 0:
        _resize_rate = int(_w / _max_window_size)
        cv2.imshow('Deleted lines and boxes', cv2.resize(frame, (int(_w / _resize_rate), int(_h / _resize_rate))))

    return frame

def add_filters(test_im, resize = 3, blur = 3, rate=2):
    """
    Adding to Pillow image filters for improving text spots recognition.
    :param test_im: Pillow image
    :param resize: value of smoothing
    :param blur: degree of box blur
    :param rate: degree of simplification
    :return: Pillow image
    """
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

def detect_fig(frame, area_treshold=0):
    """
    OpenCV recognition of rectangles around text spots.

    :param frame: OpenCV image
    :param area_treshold: minimum value of rectangle (box) area which were extracted
    :return: results of recognition (OpenCV image), boxes' list
    """
    boxes = []
    count=0
    _h, _w, _c = frame.shape
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_img = (255 - gray_img)
    thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] # применяем цветовой фильтр
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # print(len(cnts))
    for c in cnts:
        cv2.drawContours(frame, [c], -1, (0, 0, 0), -1)


    # Draw rectangles, the 'area_treshold' value was determined empirically
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # print(len(cnts))

    for c in cnts:
        if cv2.contourArea(c) > area_treshold:
            x, y, w, h = cv2.boundingRect(c)
            # print(x, y, w, h)
            boxes.append((x, y, w, h))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (36, 255, 12), 3)

    return frame, boxes

def filter_interceptions(boxes):
    for box1 in boxes:
        for box2 in boxes:
            if box1 == box2:
                continue
            ans, inner = overlap(box1, box2)
            if inner!=None:
                boxes.remove(inner)
            elif ans == True:
                boxes.remove(box2)
    return boxes

def get_rows(boxes_img):
    """Extracting rows from PIL images list"""

    tables = []
    cells = []
    for box in boxes_img:
        imageCopy = box.copy()
        cleared_box = convert_to_pil(clear_lines_and_boxes(np.array(box),80,80, _max_window_size=0))
        imageCopy = convert_to_cv(add_filters(cleared_box, resize=1, blur=1, rate=1))
        # cv2.imshow('Result', imageCopy)
        imageCopy, boxes = detect_fig(imageCopy, area_treshold=box.size[0]*0.01)
        rows = merge_in_rows(boxes)
        # area_filter = lambda row: row[-1]*row[-2]<100
        # rows = list(filter(area_filter, rows))
        print('Rows detected:', len(rows))
        # boxes_filtred = filter_interceptions(boxes)
        # print('After filtering:', len(boxes_filtred))
        for row in rows:
            x, y, w, h = row
            if w * h > 100:
                cv2.rectangle(imageCopy, (x, y), (x + w, y + h), (99, 99, 100), 3)
            else:
                rows.remove(row)
        # im_show([box,imageCopy], ncol=1)
        if len(rows)>=2:
            _cleared_box = np.array(cleared_box)
            for box in boxes:
                x, y, w, h = box
                cv2.rectangle(_cleared_box, (x,y), (x+w, y+h), (36, 255, 12), 3)
            for row in rows:
                x, y, w, h = row
                cv2.rectangle(_cleared_box, (x,y), (x+w, y+h), (0, 99, 100), 3)

            # tables.append((convert_to_pil(_cleared_box), rows))
            tables.append((cleared_box, rows))
        elif len(rows)==1:
            cells.append((cleared_box, rows[0]))
    return tables, cells

if __name__=='__main__':
    #Step 0: Extract images from PDF:
    convert_pdf2png()

    #Step 1: Load extracted image with Pillow
    image_path = r'./tmp/page0.jpg'
    img = Image.open(image_path) #pil image
    _img = np.array(img) #cv image

    # Defining size of image for displaying with OpenCV
    _h, _w, _c = _img.shape
    _resize_rate = int(_w / 600)

    cv2.imshow('Original', cv2.resize(_img, (int(_w / _resize_rate), int(_h / _resize_rate))))

    #Step 2: Extracting columns
    # Detecting spots of text.
    _img = clear_lines_and_boxes(_img, 500, 40)
    # cv2.imshow('Clearing', cv2.resize(_img, (int(_w / _resize_rate), int(_h / _resize_rate))))
    img = convert_to_pil(_img)

    imageCopy = img.copy()
    imageCopy = convert_to_cv(add_filters(imageCopy))
    imageCopy, boxes = detect_fig(imageCopy)
    cv2.imshow('Result', cv2.resize(imageCopy, (int(_w / _resize_rate), int(_h / _resize_rate))))

    groups_of_boxes = group_boxes_to_axis(boxes, axis=1, margin=50)
    # group_images = crop_image_to_boxes(img, groups_of_boxes[0])


    #Step 3: Extract and show text boxes
    boxes_img = []
    for box in boxes:
        x, y, w, h = box
        width, height = img.size
        box_img = crop_image(img, x, y, x + w, y + h)
        boxes_img.append(box_img)

    im_show(boxes_img)

    tables = []
    table_rows = []

    # Splits box to rows:
    # Here we are using filter to get spots of text separated with spaces for choosing grouped in columns text
    # Parameters for filter was picked up with config_filter.py
    for box in boxes_img:
        imageCopy = box.copy()
        cleared_box = convert_to_pil(clear_lines_and_boxes(np.array(box),80,80, _max_window_size=0))
        imageCopy = convert_to_cv(add_filters(cleared_box, resize=1, blur=1, rate=1))
        imageCopy, boxes = detect_fig(imageCopy, area_treshold=box.size[0]*0.01)
        rows = merge_in_rows(boxes)
        print('Rows detected:', len(rows))
        for row in rows:
            x, y, w, h = row
            if w * h > 100:
                cv2.rectangle(imageCopy, (x, y), (x + w, y + h), (99, 99, 100), 3)
            else:
                rows.remove(row)
        # im_show([box,imageCopy], ncol=1)
        if len(rows)>=2:
            _cleared_box = np.array(cleared_box)
            for box in boxes:
                x, y, w, h = box
                cv2.rectangle(_cleared_box, (x,y), (x+w, y+h), (36, 255, 12), 3)
            for row in rows:
                x, y, w, h = row
                cv2.rectangle(_cleared_box, (x,y), (x+w, y+h), (0, 99, 100), 3)

            tables.append(convert_to_pil(_cleared_box))
            table_rows.append(table_rows)

    print(len(tables))
    if len(tables)>0:
        im_show(tables, ncol=11)
    #Tables list is result.
    #Type of images in table list: Pillow
    # Use it for further text recognition.
