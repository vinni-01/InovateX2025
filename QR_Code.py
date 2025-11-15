import cv2
import os

folder = "sources/"

for i in os.listdir(folder):
	if i.lower().endswith((".jpg")):
		img_path = os.path.join(folder, i)
		img = cv2.imread(img_path)
		detector = cv2.QRCodeDetector()

		data, bbox, _ = detector.detectAndDecode(img)



		if bbox is not None and data != "":
			for j in bbox:
				box = j.astype(int)
				for i in range(len(box)):
					cv2.line(img, tuple(box[i]), tuple(box[(i + 1) % len(box)]), (255, 0, 0), 3)
		cv2.imshow("QR Detected", img)

		cv2.waitKey(0)
		cv2.destroyWindow("QR Detected")
