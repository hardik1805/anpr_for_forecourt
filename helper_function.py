import numpy as np
import cv2
import time
import easyocr
import pygame
import requests
import json
import string
import re
import os

reader = easyocr.Reader(['en'])
global min_threshold
min_threshold = 0.5

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5',
                    'Z': '2',
                    'B': '8'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '6': 'G',
                    '5': 'S',
                    '8': 'B'}


def license_complies_format(text):
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
            (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
            (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
            (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
            (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
            (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
            (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False


def format_license(text):
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_


def contains_letters_and_numbers(text):
    # Check if the text contains at least one letter and one number
    return bool(re.search(r'[A-Za-z]+.*\d+|\d+.*[A-Za-z]+', text))


def get_text_recognizer_api(region, text):
    if not os.path.exists('temp'):
        os.mkdir('temp')
    licence_image_path = f'temp/{text}.jpg'
    cv2.imwrite(licence_image_path, region)

    with open(licence_image_path, 'rb') as fp:
        licence_response = requests.post(
            'https://api.platerecognizer.com/v1/plate-reader/',
            files=dict(upload=fp),
            data=dict(region=['gb']),
            headers={'Authorization': 'Token 1bec968c132616e3a8dae8f1dbbbf73a2b2e4015'})

    fp.close()
    os.remove(licence_image_path)

    result = licence_response.json()['results']
    if len(result) == 0:
        return {'text': '', 'score': 0.0}
    result = result[0]
    text, score = result['plate'].upper(), result['score']
    if license_complies_format(text):
        text = format_license(text)
    return [text, score]


def read_text(threshold_image, region_threshold=0.5, car_region=False):
    ocr_result = reader.readtext(threshold_image)
    rectangle_size = threshold_image.shape[0] * threshold_image.shape[1]
    plate = []
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        if length * height / rectangle_size > region_threshold:
            text = ''.join(e for e in result[1] if e.isalnum()).upper()
            if len(text) > 7:
                continue
            if license_complies_format(text):
                text = format_license(text)
            plate.append([text, result[2]])

    if len(plate) == 0:
        return ['', 0.0]
    elif len(plate) == 1:
        return plate[0]
    else:
        for text, score in plate:
            if contains_letters_and_numbers(text):
                return text, score


def plate_validation(ocr_text, rec_text):
    if len(ocr_text) == len(rec_text) or len(ocr_text) < len(rec_text):
        return rec_text
    elif len(ocr_text) > len(rec_text):
        return ocr_text


def create_image_with_text(text, img_width, img_height):
    # Define the font properties
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_thickness = 2
    text_color = (0, 0, 0)  # Black color in BGR format
    # Get the text size
    text_size, _ = cv2.getTextSize(text, font_face, font_scale, font_thickness)
    # Set the image size based on the text size
    image_width = img_width + 10
    image_height = img_height + 10
    # Create a white image
    image = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255
    # Calculate the position to center the text
    text_x = int((image_width - text_size[0]) / 2)
    text_y = int((image_height + text_size[1]) / 2)

    # Put the text on the image
    cv2.putText(
        image,
        text,
        (text_x, text_y),
        font_face,
        font_scale,
        text_color,
        font_thickness,
        cv2.LINE_AA,
    )

    return image


def show_licence_text(img, boxes, plate_text):
    y1, x1, y2, x2 = map(int, boxes[:4])
    y, x, h, w = x1, y1, (x2 - x1), (y2 - y1 + 50)
    h, w = img[y:y + h, x:x + w].shape[:2]
    text_img_resized = cv2.resize(create_image_with_text(plate_text, w, h), (w, h))
    img[y - h:y, x:x + w] = text_img_resized
    return img


def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2 = license_plate
    found_car = None
    for car_id, (x_car1, y_car1, x_car2, y_car2, _) in enumerate(vehicle_track_ids):
        if x1 > x_car1 and y1 > y_car1 and x2 < x_car2 and y2 < y_car2:
            found_car = vehicle_track_ids[car_id]
            break
    if found_car is not None:
        return list(map(int, found_car))
    return [-1, -1, -1, -1, -1]


def seconds_to_hms(seconds):
    # Calculate hours, minutes, and remaining seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60
    print(f"Elapsed time: {hours} hours, {minutes} minutes, {seconds} seconds")
    return hours, minutes, remaining_seconds


def play_music():
    file_path = "beep.mp3"
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()


def get_registration_info(license_number):
    start_time = time.time()
    print('Checking number plates with DVLA:', license_number, end="")
    url = "https://driver-vehicle-licensing.api.gov.uk/vehicle-enquiry/v1/vehicles"

    payload = "{\n\t\"registrationNumber\": \"" + license_number + "\"\n}"
    headers = {
        'x-api-key': 'KGcORPSbWp1cyY4yE5XA34CbwPDvF5Ud4h973ijY',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    # print(response.json())
    print(f' Done!! Took {time.time() - start_time} seconds')
    return {"status_code": response.status_code, "response": json.loads(response.text)}


# get_registration_info("LN66VJA")
# get_registration_info("KR20NXB")


# def transform_number_plate(original_plate):
#     if len(original_plate) == 7:
#         first_two_chars = original_plate[:2]
#         third_char = original_plate[2]
#         forth_char = original_plate[3]
#         last_chars = original_plate[4:]
#
#         if first_two_chars.isalpha() and last_chars.isalpha() and (
#                 (third_char == 'Z' and forth_char.isdigit()) or
#                 (third_char.isdigit() and forth_char == 'Z')
#         ):
#             transformed_plate = first_two_chars + ('7' if third_char == 'Z' else third_char) + (
#                 '7' if forth_char == 'Z' else forth_char) + last_chars
#             return transformed_plate
#

#     return original_plate
