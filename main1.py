import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
import os
from datetime import datetime
import xlwings as xw

# Initialize Models
plate_detector = YOLO("best.pt")  # Your trained model
ocr = PaddleOCR(use_angle_cls=True, lang='en', rec_algorithm='SVTR_LCNet')

# Excel Setup
def init_excel():
    if not os.path.exists("plate_logs.xlsx"):
        wb = xw.Book()
        ws = wb.sheets[0]
        ws.range("A1").value = ["Plate Number", "Date", "Time"]
        wb.save("plate_logs.xlsx")
        wb.close()
    
    return xw.Book("plate_logs.xlsx")

# Enhanced OCR Preprocessing
def preprocess_plate(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((1,1), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

def recognize_plate(frame):
    # Detection
    results = plate_detector(frame)
    plates = []
    
    for box in results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0:
            continue
            
        # Preprocess and OCR
        processed = preprocess_plate(crop)
        ocr_result = ocr.ocr(processed, cls=True)
        
        if ocr_result and ocr_result[0]:
            plate_text = " ".join([res[1][0] for res in ocr_result[0] if res[1][1] > 0.6])
            if plate_text:
                plates.append((plate_text, (x1, y1, x2, y2)))
    
    return plates

# Main Processing
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    wb = init_excel()
    ws = wb.sheets[0]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        plates = recognize_plate(frame)
        
        for plate_text, (x1, y1, x2, y2) in plates:
            # Save to Excel
            last_row = ws.range("A" + str(ws.cells.last_cell.row)).end('up').row + 1
            ws.range(f"A{last_row}").value = [
                plate_text,
                datetime.now().strftime('%Y-%m-%d'),
                datetime.now().strftime('%H:%M:%S')
            ]
            
            # Visualize
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, plate_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        
        cv2.imshow("Plate Recognition", frame)
        if cv2.waitKey(1) == 27:
            break
    
    wb.save()
    wb.close()
    cap.release()
    cv2.destroyAllWindows()

# Run Test
if __name__ == "__main__":
    process_video("final.mp4")  # Replace with your test video