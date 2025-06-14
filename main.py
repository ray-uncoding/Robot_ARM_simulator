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
    print("錯誤：無法開啟攝影機。手勢控制將被禁用。")

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

# Main Application Loop
plt.ion()
fig.show()

print("啟動主迴圈。在 OpenCV 視窗中按下 'q' 鍵，或關閉 Matplotlib 視窗以退出。")

try:
    while True:
        frame_for_display = None

        # Gesture Recognition
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # The new get_command returns joint, action, value, and annotated_image
                selected_joint, action, value, annotated_image = gesture_recognizer.get_command(frame)
                frame_for_display = annotated_image
                
                # selected_joint_str, action_str = gesture_recognizer.get_command() # Old call

                if selected_joint is not None:
                    try:
                        joint_idx = selected_joint - 1 # selected_joint is already an int or None
                        if 0 <= joint_idx < 6:
                            current_slider = sliders[joint_idx]
                            current_angle = current_slider.val
                            new_angle = current_angle

                            if gesture_recognizer.control_mode == "relative_continuous" and action == "set_angle_continuous":
                                angle_change = value * np.pi * 2 
                                new_angle += angle_change * gesture_recognizer.angle_step_continuous 
                            
                            elif gesture_recognizer.control_mode == "absolute" and action == "set_angle_absolute":
                                # 'value' from gesture_api is a normalized Y position (0 to 1 within active zone)
                                # Map this normalized value to the slider's min and max range
                                slider_min = current_slider.valmin
                                slider_max = current_slider.valmax
                                # We want hand moving UP (smaller Y, smaller normalized_y) to correspond to slider_max (or min, depending on convention)
                                # Current: normalized_y = 0 for top of active zone, 1 for bottom.
                                # Let's map: top of active zone (normalized_y=0) -> slider_max, bottom (normalized_y=1) -> slider_min
                                new_angle = slider_max - (value * (slider_max - slider_min))
                                # Or, if hand up = min angle: new_angle = slider_min + (value * (slider_max - slider_min))

                            elif gesture_recognizer.control_mode == "discrete":
                                if action == "increase":
                                    new_angle += DISCRETE_ANGLE_STEP
                                elif action == "decrease":
                                    new_angle -= DISCRETE_ANGLE_STEP
                            
                            new_angle = np.clip(new_angle, current_slider.valmin, current_slider.valmax)
                            
                            if abs(new_angle - current_angle) > 1e-5: # Check if there's a change
                                current_slider.set_val(new_angle)
                                # vis.draw_arm(np.array([s.val for s in sliders])) # Already handled by slider.on_changed

                    except TypeError: # Catches if selected_joint is None and we try to use it as index
                        # print(f"Debug: Selected joint is None or not an int: {selected_joint}")
                        pass 
                    except IndexError:
                        # print(f"Debug: Joint index out of range: {joint_idx}")
                        pass
            else:
                # print("無法接收影像幀（影像流結束？）。") # Keep if useful, or remove if too noisy
                pass # Continue running, sliders will still work

        # Display Camera Feed
        if frame_for_display is not None:
            cv2.imshow('手勢控制 (Gesture Control)', frame_for_display)
        elif cap.isOpened(): # Placeholder if camera is open but frame processing failed
            placeholder_frame = np.zeros((gesture_recognizer.h, gesture_recognizer.w, 3), dtype=np.uint8)
            cv2.putText(placeholder_frame, "No camera feed / Error", (50, gesture_recognizer.h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('手勢控制 (Gesture Control)', placeholder_frame)

        # Matplotlib Update
        plt.pause(0.01)

        # Exit Conditions
        quit_key_pressed = False
        if cv2.getWindowProperty('手勢控制 (Gesture Control)', cv2.WND_PROP_VISIBLE) >= 1:
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                quit_key_pressed = True
        
        if quit_key_pressed or not plt.fignum_exists(fig.number):
            print("偵測到退出指令，準備關閉...")
            break
            
except KeyboardInterrupt:
    print("偵測到使用者中斷 (Ctrl+C)。")
finally:
    print("正在清理資源...")
    # Gesture Recognizer Cleanup
    if cap.isOpened():
        gesture_recognizer.release()
        cap.release()
    cv2.destroyAllWindows()
    if plt.fignum_exists(fig.number):
         plt.close(fig)
    plt.ioff()
    gesture_recognizer.release() # Ensure mediapipe resources are released
    print("程式已結束。")
