import numpy as np
from kinematics import SixAxisArmKinematics
from visualization import ArmVisualizer
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import cv2
from gesture_api import GestureRecognizer

# DH Parameters
dh_params = [
    (0, 0.3, 0, np.pi/2), (0, 0, 0.6, 0), (0, 0, 0.5, 0),
    (0, 0.3, 0, np.pi/2), (0, 0, 0, -np.pi/2), (0, 0.15, 0, 0)
]

# Link and Joint Radii
link_radii = [0.1, 0.08, 0.06, 0.04, 0.04, 0.03]
joint_axis_radii = [0.025, 0.06, 0.06, 0.025, 0.025, 0.025]

# Kinematics and Visualization Initialization
kin = SixAxisArmKinematics(dh_params)
vis = ArmVisualizer(dh_params, link_radii=link_radii, joint_axis_radii=joint_axis_radii)

# Initial Joint Angles
init_angles = np.zeros(6)
init_angles[1] = np.pi / 2
# current_joint_angles = init_angles.copy() # This variable is not directly used later, sliders hold current state

# Gesture Control Initialization
# Parameters for GestureRecognizer can be tuned here:
SENSITIVITY_THRESHOLD = 15  # Pixels, for deadzone in relative continuous control
ANGLE_STEP_CONTINUOUS = 0.4 # Normalized angle step for relative continuous control (was 1.5, then 0.15)
CONTROL_MODE = "absolute" # Changed from "continuous" to "absolute"

gesture_recognizer = GestureRecognizer(
    sensitivity_threshold=SENSITIVITY_THRESHOLD, 
    angle_step_continuous=ANGLE_STEP_CONTINUOUS, 
    control_mode=CONTROL_MODE
)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open camera. Gesture control will be disabled.")

# ANGLE_ADJUSTMENT_STEP = np.deg2rad(2.5)  # Degrees per gesture command (used for discrete mode)
DISCRETE_ANGLE_STEP = np.deg2rad(2.5) # For discrete mode, if used

# Matplotlib Sliders Setup
fig = vis.fig
axcolor = 'lightgoldenrodyellow'
sliders = []
for i in range(6):
    ax_slider = plt.axes([0.15, 0.02 + i * 0.04, 0.65, 0.03], facecolor=axcolor)
    valmin, valmax = (0, np.pi) if i == 1 else (-np.pi, np.pi)
    slider = Slider(ax_slider, f'Joint {i+1}', valmin, valmax, valinit=init_angles[i])
    sliders.append(slider)

def update_arm_visualization(val): # Renamed for clarity
    angles = np.array([s.val for s in sliders])
    vis.draw_arm(angles)

for s in sliders:
    s.on_changed(update_arm_visualization)

# Initial Arm Drawing
vis.draw_arm(init_angles)

# Helper functions for angle calculation
def _calculate_new_angle_relative_continuous(current_angle, control_value, angle_step_factor, full_circle_radians=np.pi * 2):
    """Calculates new angle for relative continuous control."""
    angle_change = control_value * full_circle_radians * angle_step_factor
    return current_angle + angle_change

def _calculate_new_angle_absolute(control_value, slider_min, slider_max):
    """Calculates new angle for absolute control.
    Assumes control_value is normalized (0 to 1).
    Maps: top of active zone (normalized_y=0) -> slider_max, bottom (normalized_y=1) -> slider_min.
    """
    return slider_max - (control_value * (slider_max - slider_min))

def _calculate_new_angle_discrete(current_angle, action, discrete_step):
    """Calculates new angle for discrete control."""
    if action == "increase":
        return current_angle + discrete_step
    elif action == "decrease":
        return current_angle - discrete_step
    return current_angle

# Function to process gestures and update arm
def handle_gesture_input(gest_recognizer, camera_capture, arm_sliders, discrete_angle_step_val):
    """Handles gesture input, calculates new joint angles, and updates sliders.
    Returns the annotated image frame for display.
    """
    frame_for_display = None
    if not camera_capture.isOpened():
        return None

    ret, frame = camera_capture.read()
    if not ret:
        # print("Unable to receive frame (stream end?).") # Or log
        return None

    selected_joint, action, value, annotated_image = gest_recognizer.get_command(frame)
    frame_for_display = annotated_image

    if selected_joint is not None and action is not None:
        try:
            joint_idx = selected_joint - 1 # selected_joint is 1-based
            if 0 <= joint_idx < len(arm_sliders):
                current_slider = arm_sliders[joint_idx]
                current_angle = current_slider.val
                new_angle = current_angle

                control_mode = gest_recognizer.control_mode
                
                if control_mode == "relative_continuous" and action == "set_angle_continuous":
                    new_angle = _calculate_new_angle_relative_continuous(current_angle, value, gest_recognizer.angle_step_continuous)
                elif control_mode == "absolute" and action == "set_angle_absolute":
                    new_angle = _calculate_new_angle_absolute(value, current_slider.valmin, current_slider.valmax)
                elif control_mode == "discrete": # Actions "increase", "decrease"
                    new_angle = _calculate_new_angle_discrete(current_angle, action, discrete_angle_step_val)

                new_angle = np.clip(new_angle, current_slider.valmin, current_slider.valmax)
                
                if abs(new_angle - current_angle) > 1e-5: # Check if there's a significant change
                    current_slider.set_val(new_angle)
        except TypeError:
            # print(f"Debug: Selected joint is None or not an int: {selected_joint}")
            pass
        except IndexError:
            # print(f"Debug: Joint index out of range: {joint_idx if 'joint_idx' in locals() else 'N/A'}")
            pass
    return frame_for_display

# Function to display frame and check for exit conditions
def display_updates_and_check_exit(figure_obj, frame_to_display, camera_capture, gest_recognizer_dims):
    """Displays the processed frame, updates matplotlib, and checks for exit conditions.
    Returns True if the program should quit, False otherwise.
    """
    if frame_to_display is not None:
        cv2.imshow('Gesture Control', frame_to_display)
    elif camera_capture.isOpened(): # Placeholder if camera is open but frame processing failed
        h_dim = gest_recognizer_dims.h
        w_dim = gest_recognizer_dims.w
        placeholder_frame = np.zeros((h_dim, w_dim, 3), dtype=np.uint8)
        cv2.putText(placeholder_frame, "No camera feed / Error", (50, h_dim // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Gesture Control', placeholder_frame)

    plt.pause(0.01) # Process Matplotlib events and update plot

    quit_key_pressed = False
    # Check if OpenCV window is visible before checking for key press
    if cv2.getWindowProperty('Gesture Control', cv2.WND_PROP_VISIBLE) >= 1:
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            quit_key_pressed = True
    
    if quit_key_pressed or not plt.fignum_exists(figure_obj.number):
        return True # Should quit
    return False # Should continue

# Main Application Loop
plt.ion()
fig.show()

print("Starting main loop. Press 'q' in the OpenCV window or close the Matplotlib window to exit.")

try:
    while True:
        # Gesture Recognition and Arm Update
        processed_frame = handle_gesture_input(gesture_recognizer, cap, sliders, DISCRETE_ANGLE_STEP)
        
        # Display Camera Feed, Matplotlib updates, and Check Exit Conditions
        if display_updates_and_check_exit(fig, processed_frame, cap, gesture_recognizer):
            print("Exit command detected, preparing to close...")
            break
            
except KeyboardInterrupt:
    print("User interrupt detected (Ctrl+C).")
finally:
    print("Cleaning up resources...")
    # Gesture Recognizer Cleanup
    if cap.isOpened():
        gesture_recognizer.release()
        cap.release()
    cv2.destroyAllWindows()
    if plt.fignum_exists(fig.number):
         plt.close(fig)
    plt.ioff()
    # gesture_recognizer.release() # Ensure mediapipe resources are released - This was duplicated, removing one.
    print("Program finished.")
