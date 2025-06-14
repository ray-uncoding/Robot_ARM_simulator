![Example GIF](img/clip0615.gif)

[查看中文文档 (View Chinese Documentation)](./README.md)

# Gesture-Based Robotic Arm Control

This project allows you to control a simulated robotic arm using hand gestures. It's designed to be beginner-friendly and easy to set up.

## Features

- Gesture control for robotic arm movement
- Real-time visualization of the arm
- Support for different control modes
- Easy setup with pip or Anaconda/Spyder
- Bilingual documentation (English and Chinese)

## Project Structure

```
Robot_ARM_simulator/
├── main.py              # Main application script
├── src/
│   ├── gesture_api.py   # Handles gesture recognition
│   ├── kinematics.py    # Calculates arm kinematics
│   └── visualization.py # Visualizes the robotic arm
├── requirements.txt     # Project dependencies
├── README.md            # Chinese documentation
└── README_en.md         # This file (English documentation)
```

## Setup and Installation

Follow these instructions to set up the project on your local machine.

### Prerequisites

- Python 3.8 or higher
- Pip (Python package installer) or Anaconda/Miniconda

### Option 1: Using Pip

1.  **Clone the repository (or download the source code):**
    ```bash
    git clone <repository_url>
    cd Robot_ARM_simulator
    ```
    (Replace `<repository_url>` with the actual URL if you are cloning it)

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Option 2: Using Anaconda/Spyder

1.  **Clone the repository (or download the source code):**
    As above.

2.  **Create a Conda environment:**
    Open Anaconda Prompt or your terminal.
    ```bash
    conda create -n robot_arm_env python=3.9  # Or your preferred Python version
    conda activate robot_arm_env
    ```

3.  **Install dependencies using pip within the Conda environment:**
    The `requirements.txt` file lists all necessary packages.
    ```bash
    pip install numpy matplotlib mediapipe opencv-python
    ```
    Alternatively, you can try installing with Conda if preferred, though pip is generally reliable for these packages:
    ```bash
    conda install numpy matplotlib
    pip install mediapipe opencv-python # mediapipe might be easier with pip
    ```

4.  **Spyder IDE (if using):**
    If you're using Spyder, ensure it's configured to use the `robot_arm_env` environment.
    - You can install Spyder into the environment: `conda install spyder`
    - Or, if using a global Spyder, configure its Python interpreter to point to the Python executable within `robot_arm_env`. (e.g., `path_to_anaconda/envs/robot_arm_env/python.exe`).
    - To install packages directly from Spyder, you can use its IPython console with `!pip install <package_name>`.

## How to Run

1.  Ensure your webcam is connected and accessible.
2.  Navigate to the project directory in your terminal (and activate the virtual environment if you created one).
3.  Run the main script:
    ```bash
    python main.py
    ```

## Usage

- Upon running `main.py`, two windows will appear:
    - An OpenCV window showing your webcam feed with hand tracking.
    - A Matplotlib window displaying the robotic arm and sliders to control its joints.
- Use your left hand to control the arm based on the implemented gestures (refer to `gesture_api.py` for details).
- The sliders in the Matplotlib window display angles in degrees.

## Code Explanation (Beginner-Friendly)

-   **`main.py`**: This is the heart of the application. It initializes the webcam, loads the hand tracking model from MediaPipe, and sets up the visualization. It continuously captures video frames, processes them to detect hand gestures using `gesture_api.py`, updates the robot arm's joint angles, and then redraws the arm using `visualization.py`. User-adjustable parameters like window sizes are at the top of this file.
-   **`gesture_api.py`**: This script is responsible for interpreting your hand movements. It takes the hand landmark data from MediaPipe and translates specific gestures (like fingers up/down, hand position) into commands to control the robot arm (e.g., select a joint, increase/decrease angle). It supports different control modes. Comments in this file (English and Chinese) explain the gesture logic.
-   **`kinematics.py`**: This file contains the mathematical formulas (forward kinematics) that calculate the position of each part of the robot arm based on its joint angles. This is how the arm's visual representation is updated.
-   **`visualization.py`**: This script uses Matplotlib to draw the robotic arm on the screen. It takes the joint angles and uses them to plot the arm's segments. It also includes the sliders that allow manual control and display of joint angles in degrees.
-   **`requirements.txt`**: This file lists all the external Python libraries needed for the project to run (e.g., `opencv-python` for camera access, `mediapipe` for hand tracking, `matplotlib` for plotting, `numpy` for numerical operations).

## Troubleshooting

-   **Webcam not found:** Ensure your webcam is properly connected and not being used by another application. You might need to change the `cv2.VideoCapture(0)` line in `main.py` if you have multiple cameras (e.g., to `cv2.VideoCapture(1)`).
-   **Dependencies installation issues:**
    -   Make sure you are in the correct virtual environment.
    -   Try upgrading pip: `pip install --upgrade pip`.
    -   Some packages might have specific OS dependencies; check their official documentation if errors persist.
-   **Slow performance:** Hand tracking can be computationally intensive. Ensure your computer meets the minimum requirements for MediaPipe. Closing other demanding applications might help.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License.
