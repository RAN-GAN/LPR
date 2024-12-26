import cv2
import sys
import os
import pytesseract
import easyocr

from PIL import Image

model_dir = "model"
# Set the path for Tesseract OCR (Windows-specific)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Constants
harcascade = "model/haarcascade_russian_plate_number.xml"
min_area = 500
output_dir = "plates"
count = 0



def easyocr_ocr(image_path):
    image = cv2.imread(image_path)
    reader = easyocr.Reader(['en'], gpu=False, model_storage_directory=model_dir)
    results = reader.readtext(image_path)
    print("Detected Texts:")
    for result in results:
        print(f"Detected Text: {result[1]} with confidence {result[2]:.2f}")
        (x1, y1), (x2, y2) = result[0][0], result[0][2]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    output_image_path = os.path.join(os.getcwd(), "output_with_text_easyocr.jpg")
    cv2.imwrite(output_image_path, image)
    print(f"Output image saved to {output_image_path}")
    
    return results


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def preprocess_image(img):
    # Convert image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Grayscale Image", img_gray)  # Display the grayscale image
    
    # Apply adaptive histogram equalization to improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_contrast = clahe.apply(img_gray)
    # cv2.imshow("Contrast Enhanced Image", img_contrast)  # Display contrast-enhanced image
    
    # Apply binary thresholding (you can adjust the threshold value as needed)
    _, img_threshold = cv2.threshold(img_contrast, 150, 255, cv2.THRESH_BINARY)
    # cv2.imshow("Thresholded Image", img_threshold)  # Display thresholded image
    
    return img_threshold

# Function to extract text from an image
def extract_text_from_image(img):
    # Preprocess the image before OCR
    img_preprocessed = preprocess_image(img)
    
    # Convert the processed image to PIL format for pytesseract
    img_pil = Image.fromarray(img_preprocessed)
    
    # Extract text using pytesseract
    text = pytesseract.image_to_string(
    img_pil, 
    lang='eng', 
    config='--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
)
    return text.strip()

# Function to process the image and detect plates
def process_image(image_path):
    global count
    
    # Load the Haar cascade
    plate_cascade = cv2.CascadeClassifier(harcascade)
    if plate_cascade.empty():
        print("Error: Haar cascade file could not be loaded.")
        return

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image. Check the file path.")
        return

    # Convert image to grayscale for detection
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect plates
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    # Process detected plates
    for (x, y, w, h) in plates:
        area = w * h
        if area > min_area:
            # Draw rectangle around detected plate
            shrink_factor = 0.9
            new_w = int(w * shrink_factor)
            new_h = int(h * shrink_factor)
            
            # Calculate the new top-left corner to keep the center
            new_x = x + int((w - new_w) / 2)
            new_y = y + int((h - new_h) / 2)
        
            cv2.rectangle(img, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 255, 0), 2)
            cv2.putText(img, "Number Plate", (new_x, new_y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            # Save and perform OCR on the detected plate region
            img_roi = img[new_y: new_y + new_h, new_x: new_x + new_w]
            plate_path = os.path.join(output_dir, f"plate_{count}.jpg")
            cv2.imwrite(plate_path, img_roi)
            print(f"Saved: {plate_path}")
            
            output_path = os.path.join(output_dir, f"ROI_{count}.jpg")
            cv2.imwrite(output_path, img_roi)
            print(f"Region of Interest saved to {output_path}")



            plate_text = easyocr(img_roi)
            print(f"Detected Text for plate {count}: {plate_text}")
            count += 1

    # Show the result
    # cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "OIP2.jpg"
    process_image(image_path)
