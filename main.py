import os
import cv2
from google.cloud import vision

PATH = "images\\b.jpg"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"


def main():
    img_content = pre_process_image(PATH)
    detect_document(img_content)


def pre_process_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image directly in grayscale

    max_height = 1024
    if image.shape[0] > max_height:
        scale_ratio = max_height / image.shape[0]
        image = cv2.resize(image, None, fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_LINEAR)

    _, encoded_image = cv2.imencode(".jpg", image)
    return encoded_image.tobytes()


def detect_document(image_content):
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_content)
    response = client.document_text_detection(image=image)

    if response.error.message:
        raise Exception(f"{response.error.message}\nFor more info on error messages, check: https://cloud.google.com/apis/design/errors")

    text = response.full_text_annotation.text
    print(text)


main()
