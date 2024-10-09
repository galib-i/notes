import os
import cv2
from google.cloud import vision

PATH = "images\\b.jpg"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"


def main():
    img_content = pre_process_image(PATH)
    detect_document(img_content)


def pre_process_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # read image directly in grayscale

    max_height = 1024
    if image.shape[0] > max_height:
        scale_ratio = max_height / image.shape[0]
        image = cv2.resize(image, None, fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_LINEAR)

    success, encoded_image = cv2.imencode(".jpg", image)

    if success:
        return encoded_image.tobytes()

    raise ValueError("Image encoding failed")

def detect_document(image_content):
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_content)
    response = client.document_text_detection(image=image)

    if response.error.message:
        raise Exception(f"{response.error.message}\nFor more info on error messages, check: https://cloud.google.com/apis/design/errors")

    if text_annotations := response.text_annotations:
        post_process_text(text_annotations)


def post_process_text(text_annotations):
    processed_words = post_process_word(text_annotations)
    print(processed_words)
    full_text = text_annotations[0].description
    lines = full_text.split('\n')

    processed_lines = []
    for line in lines:
        if line and line[0].islower() and processed_lines:
            processed_lines[-1] += f' {line}'
        else:
            processed_lines.append(line)
    #print(processed_lines)
    final_text = '\n'.join(processed_lines)
    #print(final_text)


def post_process_word(text_annotations):
    words_with_coords = []

    for annotation in text_annotations[1:]:  # skip the first annotation as it is the full text
        word = annotation.description
        vertices = annotation.bounding_poly.vertices
        top_left_x = vertices[0].x
        top_left_y = vertices[0].y

        word = f"{word} ({top_left_x}, {top_left_y})"

        words_with_coords.append(word)

    return words_with_coords

main()