import cv2
import numpy as np
import easyocr
import os

# Constants
model_dir = "model"
min_area = 500
output_dir = "plates"

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'], gpu=False, model_storage_directory=model_dir)

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Display and save intermediate images
def save_and_show_image(stage, img):
    filename = os.path.join(output_dir, f"{stage}.jpg")
    cv2.imwrite(filename, img)
    cv2.imshow(stage, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Preprocess image (enhanced for license plates)
def preprocess_image(img):
    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Use adaptive thresholding for better contrast in varying lighting conditions
    img_thresh = cv2.adaptiveThreshold(
        img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Morphological operations to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_morph = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)

    return img_morph

# Correct perspective of a region
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

# Perform OCR using EasyOCR
def easyocr_ocr(image):
    results = reader.readtext(image)
    return results

# Detect plate using contours with better filtering
def detect_plate_using_contours(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image, 100, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    plates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:  # Likely a rectangle
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / h
                if 2 < aspect_ratio < 5:  # Typical license plate dimensions
                    plates.append(approx)
    return plates

# Process image and detect plates
def process_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image. Check the file path.")
        return

    save_and_show_image("Original_Image", img)

    # Preprocess the image
    preprocessed_img = preprocess_image(img)
    save_and_show_image("Preprocessed_Image", preprocessed_img)

    # Detect plates using contours
    plates = detect_plate_using_contours(preprocessed_img)

    best_confidence = 0
    best_plate_text = None
    best_plate_img = None

    for plate in plates:
        warped = correct_perspective(img, plate)
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, plate_binary = cv2.threshold(warped_gray, 128, 255, cv2.THRESH_BINARY)

        ocr_results = easyocr_ocr(plate_binary)

        for result in ocr_results:
            text, confidence = result[1], result[2]
            if confidence > best_confidence:
                best_confidence = confidence
                best_plate_text = text
                best_plate_img = warped

    if best_plate_img is not None:
        print(f"Best Plate Text: {best_plate_text} with confidence {best_confidence:.2f}")
        save_and_show_image("Best_Plate", best_plate_img)
    else:
        print("No plates detected with sufficient confidence.")

if __name__ == "__main__":
    image_path = "car4.jpg"  # Replace with your image file path
    process_image(image_path)
