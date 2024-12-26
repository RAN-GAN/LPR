import cv2
import os
import easyocr

# Constants
model_dir = "model"
harcascade = "model/haarcascade_russian_plate_number.xml"
min_area = 500
output_dir = "plates"
count = 0

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'], gpu=False, model_storage_directory=model_dir)

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Display and save intermediate images
def save_and_show_image(stage, img, count=None):
    if count is not None:
        filename = os.path.join(output_dir, f"{stage}_{count}.jpg")
    else:
        filename = os.path.join(output_dir, f"{stage}.jpg")
    cv2.imwrite(filename, img)
    cv2.imshow(stage, img)
    cv2.waitKey(0)  # Press any key to close the image window
    cv2.destroyAllWindows()

# Preprocess image (grayscale, contrast enhancement)
def preprocess_image(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # save_and_show_image("Gray_Image", img_gray)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_contrast = clahe.apply(img_gray)
    # save_and_show_image("Contrast_Enhanced", img_contrast)

    _, img_threshold = cv2.threshold(img_contrast, 150, 255, cv2.THRESH_BINARY)
    # save_and_show_image("Thresholded_Image", img_threshold)

    return img_threshold

# Perform OCR using EasyOCR
def easyocr_ocr(image):
    results = reader.readtext(image)
    print("Detected Texts:")
    for result in results:
        print(f"Detected Text: {result[1]} with confidence {result[2]:.2f}")
    return results

# Process image and detect plates
def process_image(image_path):
    global count

    # Load Haar cascade
    plate_cascade = cv2.CascadeClassifier(harcascade)
    if plate_cascade.empty():
        print("Error: Haar cascade file could not be loaded.")
        return

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image. Check the file path.")
        return
    save_and_show_image("Original_Image", img)

    # Convert image to grayscale for detection
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # img_contrast = clahe.apply(img_gray)
    image = preprocess_image(img)

    # Detect plates
    plates = plate_cascade.detectMultiScale(image, 1.1, 4)

    for (x, y, w, h) in plates:
        area = w * h
        if area > min_area:
            # Draw rectangle around detected plate
            shrink_factor = 0.9
            new_w = int(w * shrink_factor)
            new_h = int(h * shrink_factor)
            new_x = x + int((w - new_w) / 2)
            new_y = y + int((h - new_h) / 2)

            cv2.rectangle(img, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 255, 0), 2)
            cv2.putText(img, "Number Plate", (new_x, new_y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            # Extract and preprocess the plate region
            img_roi = img[new_y: new_y + new_h, new_x: new_x + new_w]
            save_and_show_image(f"Detected_Plate_{count}", img_roi, count)

            preprocessed_img = preprocess_image(img_roi)

            # Perform OCR
            plate_text = easyocr_ocr(preprocessed_img)
            print(f"Detected Text for plate {count}: {plate_text}")
            count += 1

    # Save the image with detections
    output_image_path = os.path.join(output_dir, "output_with_detections.jpg")
    cv2.imwrite(output_image_path, img)
    save_and_show_image("Final_Output", img)

if __name__ == "__main__":
    image_path = "car4.jpg"  # Replace with your image file path
    process_image(image_path)
