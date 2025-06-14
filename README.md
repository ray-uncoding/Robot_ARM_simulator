# Gesture-Based Robotic Arm Simulator

This project allows you to control a simulated 6-axis robotic arm using hand gestures detected by your webcam. It features multiple control modes and provides real-time visual feedback.

## Features

*   **Real-time Gesture Control:** Control a 6-axis robotic arm using hand gestures.
*   **Multiple Control Modes:**
    *   **Relative Continuous Mode:** Adjust the selected joint's angle by moving your left hand up or down.
    *   **Absolute Mode:** Directly map your left hand's vertical position within a defined control area to the selected joint's angle.
    *   **Discrete Mode:** Use specific left-hand gestures (fist/thumbs-up for increase, index finger up for decrease, open palm to stop) to change the selected joint's angle.
*   **Dynamic Instructions:** On-screen instructions adapt to the selected control mode.
*   **Visual Feedback:** See your hand landmarks and the robotic arm's movement in real-time.
*   **Interactive Sliders:** Manually control each joint using Matplotlib sliders.
*   **Modular Code:** The project is structured for better readability and maintainability.

## Prerequisites

*   Python 3.7+ (Anaconda distribution recommended for beginners)
*   A webcam connected to your computer.

## Installation (General Python / pip)

These instructions are for users familiar with standard Python environments and `pip`. If you are new or prefer Anaconda, see the next section.

1.  **Download or Clone the Project:**
    *   **Clone (if you have git):**
        ```bash
        git clone <repository_url> 
        cd Robot_ARM_simulator
        ```
    *   **Download ZIP:** Download the project files as a ZIP from the repository page and extract them to a folder on your computer.

2.  **Create a Virtual Environment (Recommended):**
    Open your terminal or command prompt in the project directory and run:
    ```bash
    python -m venv venv
    ```
    Activate the virtual environment:
    *   On Windows (PowerShell/CMD):
        ```powershell
        .\venv\Scripts\Activate.ps1 
        ```
        or
        ```cmd
        venv\Scripts\activate.bat
        ```
    *   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

3.  **Install Dependencies:**
    With the virtual environment activated, install the required Python packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
    This will install `numpy`, `matplotlib`, `mediapipe`, and `opencv-python`.

## Alternative Setup for Anaconda Users (Spyder IDE)

This guide is for beginners or those who prefer using Anaconda and the Spyder IDE.

1.  **Download and Install Anaconda:**
    *   Go to the [Anaconda Distribution page](https://www.anaconda.com/products/distribution).
    *   Download the installer for your operating system (Windows, macOS, or Linux).
    *   Run the installer and follow the on-screen instructions. It's usually fine to accept the default settings.

2.  **Download the Project Files:**
    *   Go to the GitHub repository page for this project.
    *   Click the "Code" button, then "Download ZIP".
    *   Save the ZIP file to your computer and then extract it to a known location (e.g., `Documents/Robot_ARM_simulator`).

3.  **Create a Conda Environment:**
    *   Open the **Anaconda Prompt** (search for it in your system's applications).
    *   Create a new environment specifically for this project. This keeps its packages separate from other Python projects. Let's call it `robot_arm_env`:
        ```bash
        conda create -n robot_arm_env python=3.9 
        ```
        (You can choose a Python version like 3.8, 3.9, or 3.10. Using 3.9 is a safe bet.)
        Press `y` and Enter when prompted to proceed.
    *   Activate the new environment:
        ```bash
        conda activate robot_arm_env
        ```
        You should see `(robot_arm_env)` at the beginning of your prompt line.

4.  **Install Dependencies in the Conda Environment:**
    *   Navigate to the project directory (where you extracted the files) in the Anaconda Prompt. For example, if you extracted it to `Documents/Robot_ARM_simulator`:
        ```bash
        cd path/to/your/Robot_ARM_simulator 
        ```
        (Replace `path/to/your/` with the actual path, e.g., `cd C:/Users/YourName/Documents/Robot_ARM_simulator`)
    *   Install the required packages using `pip` (which is available within your conda environment) and the `requirements.txt` file:
        ```bash
        pip install -r requirements.txt
        ```
        This will install `numpy`, `matplotlib`, `mediapipe`, and `opencv-python` into your `robot_arm_env`.

5.  **Launch Spyder and Open the Project:**
    *   In the same Anaconda Prompt (with `robot_arm_env` activated), type:
        ```bash
        spyder
        ```
        This will launch the Spyder IDE. It might take a moment the first time.
    *   In Spyder, go to "File" > "Open..." and navigate to the `Robot_ARM_simulator` folder. Select the `main.py` file and click "Open".
    *   You can also use Spyder's "Projects" feature: "Projects" > "New Project..." > "Existing Directory", and select your `Robot_ARM_simulator` folder.

6.  **Run the Simulator in Spyder:**
    *   With `main.py` open and active in Spyder, click the green "Run file" button (it looks like a play icon) in the toolbar, or press `F5`.
    *   Two windows should appear: one is the Matplotlib window showing the 3D robotic arm, and the other is the OpenCV window showing your webcam feed with gesture instructions.

## Usage

1.  **Start the Application:**
    *   **If using general Python/pip:** Navigate to the project directory in your terminal (with the virtual environment activated) and run:
        ```bash
        python main.py
        ```
    *   **If using Anaconda/Spyder:** Run `main.py` from within Spyder as described above.

2.  **Understanding the Windows:**
    *   **Robotic Arm Window (Matplotlib):** Shows a 3D visualization of the 6-axis robotic arm. You can use the sliders at the bottom to manually control each joint.
    *   **Camera Feed & Instructions Window (OpenCV):** Shows your webcam feed. This is where your hand gestures are detected. Instructions for the current control mode are displayed here.

3.  **Controls:**
    *   **Camera:** The application will use your default webcam. Ensure it's not covered and has adequate lighting.
    *   **Right Hand (Joint Selection):** Your right hand selects which joint of the robotic arm to control.
        *   **Gesture '1' (Index Finger Up):** Control Joint 1
        *   **Gesture '2' (Index & Middle Finger Up - Peace Sign):** Control Joint 2
        *   **Gesture '3' (Index, Middle, Ring Finger Up):** Control Joint 3
        *   **Gesture '4' (Index, Middle, Ring, Pinky Finger Up):** Control Joint 4
        *   **Gesture '5' (Open Palm - Five Fingers Up):** Control Joint 5
        *   **Gesture '0' (Fist):** Control Joint 6
    *   **Left Hand (Angle Adjustment):** Your left hand adjusts the angle of the *selected* joint, based on the current control mode.
        *   **Control Point:** The `MIDDLE_FINGER_MCP` (the knuckle of your middle finger) is used as the primary control point for your left hand.

4.  **Switching Control Modes:**
    *   Press **'m'** on your keyboard (while the OpenCV window is active) to cycle through the different control modes:
        1.  `relative_continuous`
        2.  `absolute`
        3.  `discrete`
    *   The current mode and specific instructions for that mode will be displayed on the OpenCV window.

5.  **Control Mode Details:**

    *   **`relative_continuous` Mode:**
        *   **To Increase Angle:** Move your left hand's `MIDDLE_FINGER_MCP` upwards.
        *   **To Decrease Angle:** Move your left hand's `MIDDLE_FINGER_MCP` downwards.
        *   The sensitivity and step size can be adjusted in `main.py` (see "Modifying the Code" section).

    *   **`absolute` Mode:**
        *   A vertical active zone is defined (though not explicitly drawn by default in recent versions to reduce clutter).
        *   Position your left hand's `MIDDLE_FINGER_MCP` vertically. The top of the camera feed corresponds to one angle limit, and the bottom to the other.
        *   The vertical position of your hand directly maps to the joint's angle.

    *   **`discrete` Mode:**
        *   **To Increase Angle:** Make a **fist** or **thumbs-up** gesture with your left hand.
        *   **To Decrease Angle:** Point your **index finger up** (like a '1') with your left hand.
        *   **To Stop Changing Angle:** Show an **open palm** with your left hand.
        *   The angle changes in predefined steps (adjustable in `main.py`).

6.  **Quitting the Application:**
    *   Press **'q'** on your keyboard (while the OpenCV window is active) to close the application.
    *   Alternatively, closing the Matplotlib window will also terminate the program.

## Understanding and Modifying the Code (For Beginners)

This project is split into several Python files. Here's a simple breakdown:

*   **`main.py`:** This is the main "brain" of the application.
    *   It sets up the robot arm, the camera, and the display windows.
    *   It contains the main loop that continuously gets gestures, updates the arm, and shows everything on screen.
    *   **Beginner-Friendly Parameters to Adjust in `main.py`:**
        *   `CONTROL_MODE`: At the top of the script, you can change the default starting mode. Options are `"absolute"`, `"relative_continuous"`, or `"discrete"`.
        *   `SENSITIVITY_THRESHOLD` (for `relative_continuous` mode): How much your hand needs to move (in pixels) before it registers as a change. Increase for less sensitivity, decrease for more.
        *   `ANGLE_STEP_CONTINUOUS` (for `relative_continuous` mode): How much the angle changes for a given hand movement. Smaller values mean finer control.
        *   `DISCRETE_ANGLE_STEP` (for `discrete` mode): How many degrees the angle changes with each "increase" or "decrease" gesture.
        *   `dh_params`: These define the robot arm's physical dimensions. Changing these will alter the arm's shape and reach. (Advanced)
        *   `init_angles`: The starting angles for each of the 6 joints.

*   **`gesture_api.py`:** This file handles all the hand tracking and gesture recognition.
    *   It uses the `mediapipe` library to find your hands and their landmarks (finger joints, palm, etc.) from the webcam image.
    *   It then interprets the positions and shapes of your hands to determine which gesture you're making (e.g., "fist", "open palm", "index finger up").
    *   It also processes the left hand's position for the different control modes.
    *   It draws the hand landmarks and instructional text onto the camera feed.

*   **`kinematics.py`:** This file contains the mathematical formulas (Denavit-Hartenberg parameters) that describe how the robotic arm is constructed and how its joints move.
    *   The `get_joint_positions()` function calculates the 3D coordinates of each joint based on the current angles. This is essential for drawing the arm.

*   **`visualization.py`:** This file is responsible for drawing the 3D robotic arm in the Matplotlib window.
    *   It takes the joint positions calculated by `kinematics.py` and draws cylinders for the links and spheres/axes for the joints.
    *   It also sets up the sliders for manual joint control.

*   **`requirements.txt`:** This is a simple text file listing the external Python packages needed for the project to run. The `pip install -r requirements.txt` command uses this file to install them.

## File Structure (Overview)

```
Robot_ARM_simulator/
├── main.py                 # Main application script
├── gesture_api.py          # Handles gesture recognition and camera feed
├── kinematics.py           # Robotic arm kinematics calculations
├── visualization.py        # 3D arm visualization using Matplotlib
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

## Troubleshooting

*   **Webcam Not Detected / "Error: Cannot open camera":**
    *   Ensure your webcam is properly connected and enabled in your system settings.
    *   Check if another application is currently using the webcam. Close it if so.
    *   In `main.py`, try changing `cap = cv2.VideoCapture(0)` to `cap = cv2.VideoCapture(1)` (or other numbers like 2, 3) if you have multiple cameras or if the default (0) isn't working.
*   **Slow Performance / Laggy Video:**
    *   Ensure your computer meets reasonable specifications for running real-time image processing.
    *   Close other resource-intensive applications.
    *   Try reducing the webcam resolution if possible (though this project currently resizes to fixed dimensions in `gesture_api.py`).
*   **Incorrect Gesture Recognition:**
    *   Ensure good, consistent lighting. Avoid very dark rooms or strong backlighting.
    *   Keep your hands clearly visible within the camera frame, not too close or too far.
    *   Avoid cluttered backgrounds that might confuse the hand detection.
    *   Make gestures clearly and distinctly.
*   **Anaconda Environment Issues:**
    *   Make sure you have *activated* the correct conda environment (`conda activate robot_arm_env`) in the Anaconda Prompt *before* running `pip install -r requirements.txt` or launching `spyder`.
    *   If Spyder seems to be using the wrong Python interpreter, you might need to configure it. In Spyder, go to "Tools" > "Preferences" > "Python interpreter" and select "Use the following Python interpreter", then point it to the Python executable within your conda environment (e.g., `C:\Users\YourName\anaconda3\envs\robot_arm_env\python.exe`).

---

This README aims to provide a comprehensive guide for users of all levels. Enjoy controlling the robotic arm!
