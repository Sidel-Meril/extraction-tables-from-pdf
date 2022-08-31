"""
Example with text recognition in cells and sorting data by key words
"""
import pdfhandler
import pytesseract
pdfhandler.poppler_path = r'./poppler-0.68.0_x86/poppler-0.68.0/bin'
pytesseract.pytesseract.tesseract_cmd  = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
page = 67

pdfhandler.convert_pdf2png(start=page,end=page)
key_word_table1 = ['Regular', 'Overtime', 'Holiday','this period', 'rate', 'hours', 'year to date']
key_word_table2 = ['Information', 'this period', 'total to date']
key_word_dates = ['Period', 'Pay Date', '/20']
def check_text_to_tables(exctracted_text, key_words) -> bool:
    """
    Checking text
    :param exctracted_text:
    :param key_words:
    :return:
    """

    for key_word in key_words:
        for text in exctracted_text:
            if key_word in text:
                return True
    return False

if __name__=='__main__':
    table1 = set()
    table2 = set()
    dates = set()
    CW = None
    image_path = fr'./tmp/page{page}.jpg'
    _pil, cv = pdfhandler.read_img(image_path)
    text = pytesseract.image_to_string(_pil).split('\n')
    CW = list(filter(lambda word: 'CW' in word, pytesseract.image_to_string(_pil).split('\n')))[0]
    cv = pdfhandler.clear_lines_and_boxes(cv, 500, 40, _max_window_size=0)
    pil = pdfhandler.convert_to_pil(cv)
    cv = pdfhandler.convert_to_cv(pdfhandler.add_filters(pil))
    _, boxes = pdfhandler.detect_fig(cv)
    groups_of_boxes = pdfhandler.group_boxes_to_axis(boxes, axis=1, margin=50)
    groups_of_boxes = list(filter(lambda boxes_set: len(boxes_set)>=2, groups_of_boxes))
    for group in groups_of_boxes:
        print('Start processing box')
        boxes_pil = pdfhandler.crop_image_to_boxes(_pil, group)
        boxes_rows, boxes_cells = pdfhandler.get_rows(boxes_pil)
        general_rows = []
        general_text = []
        for pair in boxes_rows:
            table_img, rows_coords = pair[0], pair[1]
            rows_coords.sort(key=lambda row: row[1])
            rows = pdfhandler.crop_image_to_boxes(pair[0], rows_coords, y_margin=8, x_margin=5)
            print('Start processing column')
            extracted_text = []
            for row in rows:
                text=pytesseract.image_to_string(row)
                extracted_text.append(text)
            general_rows.extend(rows)
            general_text.extend(extracted_text)
            if check_text_to_tables(extracted_text, key_word_table1):
                table1.add(tuple(extracted_text))
            if check_text_to_tables(extracted_text, key_word_table2):
                table2.add(tuple(extracted_text))
            if check_text_to_tables(extracted_text, key_word_dates):
                dates.add(tuple(extracted_text))
        if len(general_rows)<=4:
            ncol=1
        else:
            ncol=4
        # Show images with extracted text:
        # try:
        #     detecting_boxes.im_show(general_rows, ncol=ncol,_titles=general_text)
        # except:
        #     print('ERROR:', len(general_rows))

    print('\nRESULTS:\n\n',table1,'\n\n',table2,'\n\n',dates,'\n\n\n', CW)
