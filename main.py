import numpy as np
from kinematics import SixAxisArmKinematics
from visualization import ArmVisualizer
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import cv2
from gesture_api import GestureRecognizer

# ==============================================================================
# ROBOT ARM CONFIGURATION (User may not need to change these frequently)
# ==============================================================================
# DH Parameters for the 6-axis robotic arm
dh_params = [
    (0, 0.3, 0, np.pi/2), (0, 0, 0.6, 0), (0, 0, 0.5, 0),
    (0, 0.3, 0, np.pi/2), (0, 0, 0, -np.pi/2), (0, 0.15, 0, 0)
]

# Visual properties for the arm links and joints
link_radii = [0.1, 0.08, 0.06, 0.04, 0.04, 0.03]
joint_axis_radii = [0.025, 0.06, 0.06, 0.025, 0.025, 0.025]

# Initial joint angles for the arm
init_angles = np.zeros(6)
init_angles[1] = np.pi / 2 # Example: Set initial angle for Joint 2 (in radians)

# ==============================================================================
# GESTURE CONTROL PARAMETERS (User can adjust these)
# ==============================================================================
# --- General Gesture Recognizer Settings ---
# Control mode for gesture input:
# "absolute": Left hand vertical position maps directly to joint angle.
# "relative_continuous": Left hand vertical movement continuously adjusts joint angle.
# "discrete": Specific left hand gestures increment/decrement joint angle.
CONTROL_MODE = "absolute"

# --- OpenCV Window Settings (User can adjust these) ---
OPENCV_WINDOW_INITIAL_WIDTH = 960
OPENCV_WINDOW_INITIAL_HEIGHT = 720

# --- Parameters for "relative_continuous" mode ---
# Sensitivity for deadzone in relative continuous control (pixels).
# Movement below this threshold will be ignored.
SENSITIVITY_THRESHOLD = 5
# Normalized angle step for relative continuous control.
# This factor scales the hand movement to angle change.
# (e.g., 0.1 means a full hand sweep might correspond to 0.1 * 2*pi radians)
ANGLE_STEP_CONTINUOUS = 0.025

# --- Parameters for "discrete" mode ---
# Angle step in radians for each "increase" or "decrease" gesture.
DISCRETE_ANGLE_STEP = np.deg2rad(4.5) # e.g., 2.5 degrees per step

# ==============================================================================
# INITIALIZATION
# ==============================================================================
# Kinematics and Visualization
kin = SixAxisArmKinematics(dh_params)
vis = ArmVisualizer(dh_params, link_radii=link_radii, joint_axis_radii=joint_axis_radii)

# Gesture Recognizer
gesture_recognizer = GestureRecognizer(
    w=OPENCV_WINDOW_INITIAL_WIDTH, # Pass initial width
    h=OPENCV_WINDOW_INITIAL_HEIGHT, # Pass initial height
    control_mode=CONTROL_MODE,
    sensitivity_threshold=SENSITIVITY_THRESHOLD,
    angle_step_continuous=ANGLE_STEP_CONTINUOUS
    # Add other parameters like absolute_control_dead_zone_v_ratio if you want to override defaults
)

# Camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera. Gesture control will be disabled.")

# ==============================================================================
# MATPLOTLIB UI SETUP
# ==============================================================================
fig = vis.fig
axcolor = 'lightgoldenrodyellow'
sliders = []
for i in range(6):
    ax_slider = plt.axes([0.15, 0.02 + i * 0.04, 0.65, 0.03], facecolor=axcolor)
    # Define angle limits for each joint slider in degrees
    # Example: Joint 2 (index 1) might have different limits
    valmin_deg, valmax_deg = (0, 180) if i == 1 else (-180, 180) # Degrees
    init_angle_deg = np.rad2deg(init_angles[i]) # Convert initial angle to degrees
    slider = Slider(ax_slider, f'Joint {i+1} (deg)', valmin_deg, valmax_deg, valinit=init_angle_deg)
    sliders.append(slider)

def update_arm_visualization(val): # val is not directly used here, reads all sliders
    """Callback function to update arm visualization when a slider value changes."""
    angles_deg = np.array([s.val for s in sliders])
    angles_rad = np.deg2rad(angles_deg) # Convert degrees to radians for visualization
    vis.draw_arm(angles_rad)

for s in sliders:
    s.on_changed(update_arm_visualization)

# Initial drawing of the arm
vis.draw_arm(init_angles) # init_angles is already in radians

# ==============================================================================
# OPENCV WINDOW SETUP
# ==============================================================================
# Create a resizable window for gesture control feed
cv2.namedWindow('Gesture Control', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Gesture Control', OPENCV_WINDOW_INITIAL_WIDTH, OPENCV_WINDOW_INITIAL_HEIGHT)

# ==============================================================================
# HELPER FUNCTIONS FOR MAIN LOOP
# ==============================================================================
def _calculate_new_angle_relative_continuous(current_angle, control_value, angle_step_factor, full_circle_radians=np.pi * 2):
    """Calculates new angle for relative continuous control."""
    angle_change = control_value * full_circle_radians * angle_step_factor
    return current_angle + angle_change

def _calculate_new_angle_absolute(control_value, slider_min, slider_max):
    """Calculates new angle for absolute control.
    Assumes control_value is normalized (0 to 1) from gesture_api.
    Mapping: control_value=0 (hand at top of active zone) -> slider_max,
             control_value=1 (hand at bottom of active zone) -> slider_min.
    """
    return slider_max - (control_value * (slider_max - slider_min))

def _calculate_new_angle_discrete(current_angle, action, discrete_step):
    """Calculates new angle for discrete control."""
    if action == "increase":
        return current_angle + discrete_step
    elif action == "decrease":
        return current_angle - discrete_step
    return current_angle

def handle_gesture_input(gest_recognizer, camera_capture, arm_sliders, discrete_angle_step_val):
    """Handles gesture input, calculates new joint angles, and updates sliders.
    Returns the annotated image frame for display, or None if no frame/camera.
    """
    frame_for_display = None
    if not camera_capture.isOpened():
        return None

    ret, frame = camera_capture.read()
    if not ret:
        return None

    selected_joint, action, value, annotated_image = gest_recognizer.get_command(frame)
    frame_for_display = annotated_image

    if selected_joint is not None and action is not None:
        try:
            joint_idx = selected_joint - 1
            if 0 <= joint_idx < len(arm_sliders):
                current_slider = arm_sliders[joint_idx]
                current_angle_deg = current_slider.val
                current_angle_rad = np.deg2rad(current_angle_deg)
                
                # Slider limits in radians for helper functions if they expect radian ranges
                slider_min_rad = np.deg2rad(current_slider.valmin)
                slider_max_rad = np.deg2rad(current_slider.valmax)
                
                new_angle_rad = current_angle_rad # Default to current angle in radians

                active_control_mode = gest_recognizer.control_mode
                discrete_angle_step_rad = discrete_angle_step_val # This is already in radians

                if active_control_mode == "relative_continuous" and action == "set_angle_continuous":
                    new_angle_rad = _calculate_new_angle_relative_continuous(
                        current_angle_rad, value, gest_recognizer.angle_step_continuous
                    )
                elif active_control_mode == "absolute" and action == "set_angle_absolute":
                    new_angle_rad = _calculate_new_angle_absolute(
                        value, slider_min_rad, slider_max_rad # Pass radian limits to helper
                    )
                elif active_control_mode == "discrete":
                    new_angle_rad = _calculate_new_angle_discrete(
                        current_angle_rad, action, discrete_angle_step_rad # discrete_angle_step_val is already in rad
                    )

                # Convert back to degrees for slider update and clipping
                new_angle_deg = np.rad2deg(new_angle_rad)
                
                # Clip the new angle to the slider's min/max limits (which are in degrees)
                new_angle_deg = np.clip(new_angle_deg, current_slider.valmin, current_slider.valmax)
                
                if abs(new_angle_deg - current_angle_deg) > 1e-3: # Threshold for floating point comparison
                    current_slider.set_val(new_angle_deg)
        except TypeError:
            pass
        except IndexError:
            pass
    return frame_for_display

def display_updates_and_check_exit(figure_obj, frame_to_display, camera_capture, gest_recognizer_instance):
    """Displays the processed frame, updates matplotlib, and checks for exit conditions.
    Returns True if the program should quit, False otherwise.
    """
    if frame_to_display is not None:
        cv2.imshow('Gesture Control', frame_to_display)
    elif camera_capture.isOpened(): # Display placeholder if camera is open but frame processing failed
        h_dim = gest_recognizer_instance.h # Use gesture recognizer's height
        w_dim = gest_recognizer_instance.w # Use gesture recognizer's width
        placeholder_frame = np.zeros((h_dim, w_dim, 3), dtype=np.uint8)
        cv2.putText(placeholder_frame, "No camera feed / Error", (50, h_dim // 2),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Gesture Control', placeholder_frame)

    plt.pause(0.01) # Process Matplotlib events and update plot

    quit_key_pressed = False
    # Check if OpenCV window is visible before checking for key press
    if cv2.getWindowProperty('Gesture Control', cv2.WND_PROP_VISIBLE) >= 1:
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            quit_key_pressed = True
    
    # Quit if 'q' is pressed or the Matplotlib window is closed
    if quit_key_pressed or not plt.fignum_exists(figure_obj.number):
        return True # Should quit
    return False # Should continue

# ==============================================================================
# MAIN APPLICATION LOOP
# ==============================================================================
plt.ion() # Turn on interactive mode for Matplotlib
fig.show()

print("Starting main loop. Press 'q' in the OpenCV window or close the Matplotlib window to exit.")

try:
    while True:
        # Handle gesture input and update arm
        processed_frame = handle_gesture_input(
            gesture_recognizer, cap, sliders, DISCRETE_ANGLE_STEP
        )
        
        # Display camera feed, update Matplotlib, and check for exit conditions
        # if display_updates_and_check_exit(fig, processed_frame, cap, gesture_recognizer):
        if display_updates_and_check_exit(fig, processed_frame, cap, gesture_recognizer): # Pass gesture_recognizer instance
            print("Exit command detected, preparing to close...")
            break
            
except KeyboardInterrupt:
    print("User interrupt detected (Ctrl+C).")
finally:
    print("Cleaning up resources...")
    if cap.isOpened():
        cap.release() # Release camera first
    # Gesture Recognizer Cleanup
    # Ensure mediapipe resources are released by calling release() on the gesture_recognizer instance
    if 'gesture_recognizer' in locals() and gesture_recognizer is not None:
        gesture_recognizer.release()
    
    cv2.destroyAllWindows()
    if plt.fignum_exists(fig.number):
         plt.close(fig)
    plt.ioff() # Turn off interactive mode
    print("Program finished.")
