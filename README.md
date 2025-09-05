# Smart Face Attendance System

This project is a desktop application for a smart face recognition-based attendance system. It uses computer vision and deep learning techniques to register users, detect faces in real-time or from group photos, and mark attendance in a database.

The application is built using Python with libraries such as TensorFlow, OpenCV, and Tkinter for the graphical user interface.

## Features

* **GUI-based Interface:** A user-friendly graphical interface built with Tkinter for easy interaction.
* **User Registration:** Register new users by entering their details and capturing face data.
* **Real-time Attendance:** Mark attendance in real-time by detecting faces from a webcam feed.
* **Group Photo Attendance:** Process a group photo to detect multiple faces and mark attendance for all recognized individuals.
* **Database Integration:** Uses an SQLite database (`attendance.db`) to store user information and attendance records.
* **Attendance Reporting:** Generate an attendance report for specific time slots, showing total classes, days present, and days absent for each student.
* **Reset Functionality:** A script (`reset_all.py`) is included to easily clear all user data and attendance records.

## Installation

### Prerequisites
* Python 3.x
* Git

### Steps

1.  **Install the required Python libraries:**
    The project relies on several libraries. You can install them using `pip`.
    ```bash
    pip install tensorflow numpy opencv-python scikit-learn
    # Additional libraries based on the code analysis:
    pip install mtcnn deepface pillow
    ```
    *Note: The project uses `tensorflow` and `mtcnn` for face detection and embedding. Ensure these are installed correctly.*

## Usage

1.  **Run the main application file:**
    ```bash
    python main.py
    ```

2.  **Register a New User:**
    * On the main screen, use the user registration section to add a new person to the system. You will need to provide a User ID and a Name.
    * The system will then capture face data to create embeddings for that user.

3.  **Take Attendance:**
    * Click on the "Take Attendance" button to start the webcam and begin real-time face detection and attendance marking.
    * You can also use the "Group Photo" option to process a static image.

4.  **View Attendance Report:**
    * Select a time slot (e.g., Morning, Afternoon, Evening) from the dropdown menu.
    * Click the "Attendance % Report" button to view a detailed report of attendance for the selected slot.

5.  **Reset the System:**
    * If you need to clear all data and start fresh, run the `reset_all.py` script.
    ```bash
    python reset_all.py
    ```
    *This will permanently delete `attendance.db`, `face_embeddings.pkl`, and all images in the `dataset` and `registered_faces` folders.*

## Project Structure

* `main.py`: The main script that runs the Tkinter GUI and orchestrates the entire application.
* `database.py`: Contains functions for interacting with the `attendance.db` SQLite database, including initializing tables, registering users, and recording attendance.
* `face_detect.py`: Script for detecting faces from a webcam or a static image using MTCNN.
* `track.py`: Handles the core logic for face recognition, comparing new face embeddings against stored ones, and marking attendance. It includes functions for cosine similarity and managing a list of marked IDs.
* `reset_all.py`: A utility script to delete all generated data files (`.db`, `.pkl`, `.csv`) and image folders, resetting the system to its initial state.
* `face_embeddings.pkl`: A serialized file that stores the face embeddings for all registered users. This file is crucial for the face recognition logic.
* `attendance.db`: The SQLite database file where user and attendance data are stored.
* `align_dataset_mtcnn.py`: A helper script (likely used in development) for aligning faces in a dataset using MTCNN before processing them.
* `face_aligner.py`: A utility class for aligning faces based on eye and mouth landmarks.
* `database.txt`: A sample or temporary file that seems to contain user data.
* `README.md`: This file, providing an overview of the project.

---

## Contributing

This is a personal project which was inspired by many project references, so feel free to fork the repository, make improvements, and submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
