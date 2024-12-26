import cv2
import numpy as np
import easyocr
import os

model_dir = "model"
min_area = 500
output_dir = "plates"
count = 0

reader = easyocr.Reader(['en'], gpu=False, model_storage_directory=model_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def save_and_show_image(stage, img, count=None):
    if count is not None:
        filename = os.path.join(output_dir, f"{stage}_{count}.jpg")
    else:
        filename = os.path.join(output_dir, f"{stage}.jpg")
    cv2.imwrite(filename, img)
    cv2.imshow(stage, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def preprocess_image(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    save_and_show_image("Gray_Image", img_gray)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_contrast = clahe.apply(img_gray)
    save_and_show_image("Contrast_Enhanced", img_contrast)

    _, img_threshold = cv2.threshold(img_contrast, 150, 255, cv2.THRESH_BINARY)
    save_and_show_image("Thresholded_Image", img_threshold)

    return img_threshold

def correct_perspective(image, box):
    rect = cv2.boundingRect(np.array(box, dtype="float32"))
    (x, y, w, h) = rect
    dst = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(np.array(box, dtype="float32"), dst)
    warped = cv2.warpPerspective(image, M, (w, h))
    return warped

def deskew_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is not None:
        for rho, theta in lines[0]:
            angle = np.rad2deg(theta) - 90
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1)
            rotated = cv2.warpAffine(image, M, (w, h))
            return rotated
    return image

def easyocr_ocr(image):
    results = reader.readtext(image)
    print("Detected Texts:")
    for result in results:
        print(f"Detected Text: {result[1]} with confidence {result[2]:.2f}")
    return results

def detect_plate_using_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    plates = []
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4: 
                plates.append(approx)
    return plates

def process_image(image_path):
    global count

    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image. Check the file path.")
        return
    save_and_show_image("Original_Image", img)

    plates = detect_plate_using_contours(img)

    for plate in plates:
        warped = correct_perspective(img, plate)
        save_and_show_image(f"Warped_Plate_{count}", warped, count)

        deskewed = deskew_image(warped)
        save_and_show_image(f"Deskewed_Plate_{count}", deskewed, count)

        preprocessed_img = preprocess_image(deskewed)

        plate_text = easyocr_ocr(preprocessed_img)
        print(f"Detected Text for plate {count}: {plate_text}")
        count += 1

    output_image_path = os.path.join(output_dir, "output_with_detections.jpg")
    cv2.imwrite(output_image_path, img)
    save_and_show_image("Final_Output", img)

if __name__ == "__main__":
    image_path = "car.jpg" 
    process_image(image_path)
