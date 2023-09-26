from ultralytics import YOLO
import cv2
from helper_function import get_car, read_text, play_music, get_registration_info, plate_validation, \
    get_text_recognizer_api
from database_management import check_blocked_list
import time
from create_excel import create_excel

lpd_model = YOLO('lpd_model.pt')  # load a custom model for license plate detection
yolo_model = YOLO('yolov8n.pt') # load a pre trained model for vehicle detection
vehicles = [2, 3, 5, 7] # class id of car, bus, truck, motorbike
yolo_model.classes = vehicles
frame_nmr = -1
results = {}
car_list = {}
final_result = {}
total_time = 0

cap = cv2.VideoCapture('/Users/hardikgangajaliya/Desktop/latest.MOV')

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

start_time = time.time()
print(start_time, 'start_time')


def get_finalize_result():
    # check licence plate if car passed from current frame
    current_car_ids = list(map(lambda x: int(x[4]), vehicle_detections_))
    for _id in car_list:
        if _id not in current_car_ids and _id not in final_result.keys():
            print('__________________________________________________________________________________________________')
            print('car_id:', _id, end=" ")
            # find text from car image using text_recognizer_api
            plate_recognise_text = get_text_recognizer_api(car_list[_id][2], car_list[_id][0])
            # compare easyocr and plate recogniser text
            final_text = plate_validation(car_list[_id][0], plate_recognise_text[0])
            # checking number plates with DVLA
            response = get_registration_info(final_text)
            # storing final result in final_result dictionary
            final_result[_id] = car_list[_id].copy()
            final_result[_id][0] = final_text

            # checking number plate in the blocked list
            blocked_result = check_blocked_list(final_text)
            print('Blocked result: ', blocked_result)

            if blocked_result is not None:
                print('Blocked vehicle detected!!')
                blocked_vehicle_info(final_result[_id][2], blocked_result["incidentdate"][:10],
                                     blocked_result["valueingbp"], final_result[_id][0])

            else:
                if response['status_code'] == 200:
                    print('Number plates is original')
                elif response['status_code'] != 200:
                    print('Number plates is duplicate or not detected properly!!')
                    print(f'{response["status_code"]} - {response["response"]["errors"][0]["title"]}')

                    if response['status_code'] == 404 or response['status_code'] == 400:
                        cv2.imshow(f'{final_result[_id][0]}-{response["response"]["errors"][0]["title"]}',
                                   final_result[_id][2])
                        play_music()


def blocked_vehicle_info(vehicle_image, date='', amount=0, text=''):
    info_str = f'Date: {date} Amount: {amount} pounds'
    cv2.putText(vehicle_image, info_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow(f'Blocked_vehicle_{text}', vehicle_image)
    play_music()
    time.sleep(1)
    play_music()
    cv2.waitKey(5000)
    cv2.destroyWindow(f'Blocked_vehicle_{text}')


while cap.isOpened():
    frame_nmr += 1

    ret, frame = cap.read()
    if not ret:
        # cv2.waitKey(0)
        cap.release()
        cv2.destroyAllWindows()
        break

    if frame_nmr % 5 == 0:
        # Copy frame to show bounding box to avoid overlapping on original frame
        region_x1, region_y1, region_x2, region_y2 = 5, 5, width, height
        cropped_img = frame[region_y1:region_y2, region_x1:region_x2]
        copy_cropped_img = cropped_img.copy()
        # cv2.rectangle(frame, (region_x1, region_y1), (region_x2, region_y2), (0, 0, 0), 5)

        try:
            # detect vehicle using yolo original model for the better accuracy of car, bus, truck, motorbike
            lpd_detections = lpd_model(cropped_img, verbose=False)[0]  # verbose=False to hide warning of yolo model
            licence_detections_ = [item for item in lpd_detections.boxes.data.tolist() if
                                   item[-2] > 0.6 and int(item[-1]) == 0]
            vehicle_detections_ = []
            if len(licence_detections_) != 0:
                yolo_detections = yolo_model.track(cropped_img, persist=True, verbose=False)[0]
                for detection in yolo_detections.boxes.data.tolist():
                    if detection[-2] > 0.6 and int(detection[-1]) in vehicles:
                        x1, y1, x2, y2, tracking_id, score, class_id = detection
                        vehicle_detections_.append([x1, y1, x2, y2, tracking_id])
                        # x1, y1, x2, y2 = map(int, detection[:4])
                        # cv2.putText(cropped_img, str(tracking_id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        # cv2.rectangle(cropped_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                for i, box in enumerate(licence_detections_):
                    x1, y1, x2, y2 = map(int, box[:4])
                    # Find the right car_id for the current license plates
                    # by checking licence plates is available inside the car bounding box
                    xcar1, ycar1, xcar2, ycar2, car_id = get_car([x1, y1, x2, y2], vehicle_detections_)

                    if car_id != -1:
                        # Create region of license plates and read text from it
                        region = copy_cropped_img[y1:y2, x1:x2]
                        license_plate_text, license_plate_text_score = read_text(region)

                        if license_plate_text and license_plate_text_score > 0.6 and len(license_plate_text) > 3:
                            #  Creating list of each car and storing its highest score and text
                            if (car_id in car_list and car_list[car_id][1] < license_plate_text_score) or (
                                    car_id in car_list and
                                    len(car_list[car_id][0]) <= len(license_plate_text)) \
                                    or car_id not in car_list:
                                car_list[car_id] = [license_plate_text, license_plate_text_score,
                                                    copy_cropped_img[ycar1:ycar2, xcar1:xcar2], region]

                                # results[car_id] = {'car_bbox': [xcar1, ycar1, xcar2, ycar2],
                                #                    'license_plate_bbox': box[:4],
                                #                    'license_plate_text': license_plate_text,
                                #                    'license_plate_text_score': license_plate_text_score,
                                #                    'license_image': region,
                                #                    'car_image': copy_cropped_img[ycar1:ycar2, xcar1:xcar2],
                                #                    'response': ''
                                #                    }
                    # Draw bounding box for license plates
                    cv2.rectangle(cropped_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                get_finalize_result()
        except Exception as e:
            print('Error occurred: ', e)

        # frame[region_y1:region_y2, region_x1:region_x2] = cropped_img

    cv2.imshow('Image', frame)
    # Press 'q' to quit the loop
    if cv2.waitKey(10) and 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

print('Total Time taken: ', time.time() - start_time)

print('Total frames are: ', frame_nmr)
create_excel(results)
