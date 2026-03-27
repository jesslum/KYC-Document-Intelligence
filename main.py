import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import uvicorn
import easyocr
from fastapi import FastAPI, UploadFile, File

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
app = FastAPI()

#Initialize the OCR Reader in English
reader = easyocr.Reader(['en'])

if not os.path.exists("output"):
    os.makedirs("output")

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

@app.post("/process-id")
async def process_id(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Kept getting Error so ENHANCE CONTRAST ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    distorted = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    #FIND LARGEST OBJECT
    contours, _ = cv2.findContours(distorted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # largest object by area
        largest_cnt = max(contours, key=cv2.contourArea)
        
        # Bounding Box coordinates
        x, y, w, h = cv2.boundingRect(largest_cnt)
        
        if w > 100 and h > 100:
            #Crop the image
            warped = img[y:y+h, x:x+w]
            
            #Resize for consistent
            warped = cv2.resize(warped, (800, 500))
            
            cv2.imwrite("output/flat_id.jpg", warped)
            
            #OCR
            results = reader.readtext(warped, detail=0) 

            return {
                "status": "Success",
                "extracted_text": results,
                "note": "Used bounding-box fallback for robust detection."
            }

    return {"status": "Error", "message": "Could not isolate the ID. Ensure the card is centered on a dark surface."}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)