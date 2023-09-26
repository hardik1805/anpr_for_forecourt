# Automatic Number Plate Recognition (ANPR) System for Forecourt



## Overview

The Automatic Number Plate Recognition (ANPR) System is a powerful tool designed to enhance security and streamline operations in forecourt management. This project leverages state-of-the-art technologies, including deep learning, optical character recognition, and database management, to provide a comprehensive solution for recognizing and monitoring vehicles entering and leaving a forecourt.


## Usage
1. Clone this repository to your local machine.
2. Install the required Python packages using `pip install -r requirements.txt`.
3. Run the ANPR system using `python main.py`.

## Features

- **License Plate Detection:** The ANPR system can accurately detect and extract license plates from vehicle images or video streams.

- **Vehicle Tracking:** It tracks the movement of vehicles within the forecourt in real-time, ensuring efficient and secure operations.

- **Text Recognition:** The system employs EasyOCR for highly accurate character recognition, enabling it to extract alphanumeric characters from license plates.

- **Integration with DVLA API:** By integrating with the Driver and Vehicle Licensing Agency (DVLA) API, the system can perform real-time checks on the legitimacy of license plates.

- **Alerting and Notifications:** The system sends alerts and notifications to forecourt attendants in case of suspicious activity, such as vehicles associated with previous drive-off or no-means payment incidents.

## Requirements

To run the ANPR System, you'll need to install the following Python packages:

```plaintext
cloudinary==1.34.0
easyocr==1.7.0
numpy==1.23.5
opencv_contrib_python==4.6.0.66
opencv_python==4.6.0.66
opencv_python_headless==4.7.0.72
openpyxl==3.0.10
pygame==2.5.2
pymongo==4.5.0
requests==2.31.0
ultralytics==8.0.157
```


## Files
- `beep.mp3`: Sound file for alerts and notifications.
- `database_management.py`: Python script for managing the ANPR system's database.
- `helper_functions.py`: Collection of helper functions used in the project.
- `lpd_model.pt`: Pre-trained YOLOv8 model for license plate detection.
- `main.py`: The main Python script to run the ANPR system.
- `README.MD`: This documentation.
- `requirements.txt`: List of required Python packages.
- `video.mov`: Sample video demonstrating the ANPR system.
- `yolov8n.pt`: Pre-trained YOLOv8 model for vehicle detection.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

### Note:

This ANPR system is a powerful tool designed to enhance security and streamline operations in forecourt management. It leverages state-of-the-art technologies, including deep learning, optical character recognition, and database management, to provide a comprehensive solution for recognizing and monitoring vehicles entering and leaving a forecourt.

For more information, please contact **Hardik Gangajaliya** at `hardikgangajaliya05@gmail.com`.