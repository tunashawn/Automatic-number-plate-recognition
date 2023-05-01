from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import sys


def paddleocr_predict(img):
    img = cv2.resize(img, (0, 0), fx=3, fy=3)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ocr = PaddleOCR(use_angle_cls=True, lang='en', det=False)
    result = ocr.ocr(gray_image)

    plate_number = ""

    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            plate_number += (line[-1][0] + " ")
    
    return plate_number


def get_plates(result, img):
    images = []
    boxes = result[0].boxes
    img = img.copy()
    for b in boxes:
        x1 = int(b.xyxy[0][0])
        y1 = int(b.xyxy[0][1])
        x2 = int(b.xyxy[0][2])
        y2 = int(b.xyxy[0][3])
        images.append(img[y1:y2, x1:x2].copy())
    return images


def get_LP_number(result, img):
    plates = get_plates(result, img)
    plate_numbers = []
    
    for plate in plates:
        number = paddleocr_predict(plate)
        plate_numbers.append(number)
    
    return plate_numbers


def draw_box(result, img):

    boxes = result[0].boxes
    plate_numbers = get_LP_number(result, img)

    for b, pnum in zip(boxes, plate_numbers):
        x1 = int(b.xyxy[0][0])
        y1 = int(b.xyxy[0][1]) - 20
        x2 = int(b.xyxy[0][2])
        y2 = int(b.xyxy[0][3])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y1 + 22), ((x2), (y1)), (0, 255, 0), -1)
        text_xy = (x1 + 2, y1 +18)
        # img, text, position, font, font_scale, color, thickness
        cv2.putText(img, pnum, text_xy, 0, 0.7, 0, 2) 
    
    return img
    
def video_draw_box(vid_path, model):
    cap = cv2.VideoCapture(vid_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output.avi', fourcc, 20, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        result = model(frame)
        frame = draw_box(result, frame)
        
        # write the flipped frame
        out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    # Release everything if job is finished
    # cap.release()
    out.release()
    cv2.destroyAllWindows()

if len(sys.argv) == 3:
    # Get weights and img_dir
    pre_trained_model = "best.pt"
    
    media_type = sys.argv[1]
    
    file_dir = sys.argv[2]
    
    # Create model
    model = YOLO(pre_trained_model)
   
    if media_type == "-image":
        img = cv2.imread(file_dir)
        result = model(img)
        img = draw_box(result, img)
        cv2.imshow("Result", img)
        cv2.waitKey(0)
        cv2.imwrite("predicted-" + file_dir, img)
        
    elif media_type == "-video":
        video_draw_box(file_dir, model)
    