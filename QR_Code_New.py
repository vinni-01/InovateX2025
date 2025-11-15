import cv2
import os
from ultralytics import YOLO

# Load your trained YOLO model
model = YOLO("runs/detect/qr_detector/weights/best.pt")

detector = cv2.QRCodeDetector()

folder = "sources/"

for fname in os.listdir(folder):
    if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(folder, fname)
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to load:", img_path)
        continue

    print("\nProcessing:", img_path)

    # Run YOLO
    results = model(img)[0]

    # Copy image for drawing
    img_vis = img.copy()

    # For each detection
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        # if you have multiple classes, check here:
        # if CLASSES[cls_id] != "qr_code": continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Crop the QR region
        crop = img[y1:y2, x1:x2]

        # Decode QR inside crop
        data, bbox, _ = detector.detectAndDecode(crop)

        if data != "":
            print(f"  QR detected (conf={conf:.2f}):", data)

            # Draw red rectangle around QR
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Optionally, put text on image
            cv2.putText(
                img_vis,
                data[:30],  # first 30 chars
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )
        else:
            print(f"  Detected box (conf={conf:.2f}) but QR decode failed")

    cv2.imshow("YOLO + QR decode", img_vis)
    key = cv2.waitKey(0)
    if key == 27:  # ESC to stop early
        break

cv2.destroyAllWindows()
