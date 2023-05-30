from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import os
import random
# cap = cv2.VideoCapture(1)  # For Webcam
# cap.set(3, 1280)
# cap.set(4, 720)
cap = cv2.VideoCapture("../potholeDetection/Videos/p1.mp4")  # For Video
model = YOLO("../Yolo-Weights/Pithole.pt")
classNames = ["pothole"]
prev_frame_time = 0
new_frame_time = 0

# Create Directory 
os.makedirs("pothole_images", exist_ok=True)
os.makedirs("pothole_coordiationes", exist_ok=True)
pothole_count = 0

def get_dummy_gps_coors():
    # Generate Random long and lat coordinates
    lat = random.uniform(-90, 90)
    lon = random.uniform(-180, 180)
    return lat, lon

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

            # Check if the confindence above of below the threshold

            if conf > 0.7:
                # Cope the Pothole image and save it 
                pothole_image = img[y1:y2, x1:x2]
                cv2.imwrite(f"pothole_images/pothole_{pothole_count}.jpg", pothole_image)
                
                # Get the dummy gps coordinates of the pothole 
                lat, lon = get_dummy_gps_coors()
                with open(f"pothole_coordiationes/pothole_{pothole_count}.txt", 'w') as f:
                    f.write(f"Latitude: {lat}, Longitude: {lon}")
                
                pothole_count += 1

    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



