import cv2
import mediapipe as mp
import math

class GestureRecognizer:
    def __init__(self, w=960, h=720, sensitivity_threshold=50, angle_step_continuous=0.01, control_mode="continuous", absolute_control_dead_zone_h_ratio=0.1, absolute_control_dead_zone_v_ratio=0.2): # Added absolute control dead zone ratios
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.8
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.w = w
        self.h = h

        self.selected_joint_internal = None
        self.action_internal = None
        self.value_internal = 0.0 # For continuous control

        self.debug_text = "Waiting for Control"
        self.debug_text_color = (0, 255, 255)

        # New state variables for relative left-hand control
        self.left_control_active = False
        self.left_initial_y = None
        self.left_initial_x = None # Added for X-axis control
        
        self.sensitivity_threshold = sensitivity_threshold # Renamed from sensitivity
        self.angle_step_continuous = angle_step_continuous # New parameter for continuous adjustment step
        self.control_mode = control_mode # "relative_continuous", "absolute", "discrete"

        # Parameters for absolute control
        self.absolute_control_dead_zone_h_ratio = absolute_control_dead_zone_h_ratio # Horizontal dead zone at center
        self.absolute_control_dead_zone_v_ratio = absolute_control_dead_zone_v_ratio # Vertical dead zone at top/bottom

    @staticmethod
    def _vector_2d_angle(v1, v2):
        v1_x, v1_y = v1
        v2_x, v2_y = v2
        try:
            angle_ = math.degrees(math.acos(
                (v1_x * v2_x + v1_y * v2_y) /
                (((v1_x ** 2 + v1_y ** 2) ** 0.5) * ((v2_x ** 2 + v2_y ** 2) ** 0.5))
            ))
        except:
            angle_ = 180
        return angle_

    @staticmethod
    def _hand_angle(hand_):
        angle_list = []
        # Thumb
        angle_list.append(GestureRecognizer._vector_2d_angle(
            (hand_[0][0]-hand_[2][0], hand_[0][1]-hand_[2][1]),
            (hand_[3][0]-hand_[4][0], hand_[3][1]-hand_[4][1])
        ))
        # Index finger
        angle_list.append(GestureRecognizer._vector_2d_angle(
            (hand_[0][0]-hand_[6][0], hand_[0][1]-hand_[6][1]),
            (hand_[7][0]-hand_[8][0], hand_[7][1]-hand_[8][1])
        ))
        # Middle finger
        angle_list.append(GestureRecognizer._vector_2d_angle(
            (hand_[0][0]-hand_[10][0], hand_[0][1]-hand_[10][1]),
            (hand_[11][0]-hand_[12][0], hand_[11][1]-hand_[12][1])
        ))
        # Ring finger
        angle_list.append(GestureRecognizer._vector_2d_angle(
            (hand_[0][0]-hand_[14][0], hand_[0][1]-hand_[14][1]),
            (hand_[15][0]-hand_[16][0], hand_[15][1]-hand_[16][1])
        ))
        # Pinky
        angle_list.append(GestureRecognizer._vector_2d_angle(
            (hand_[0][0]-hand_[18][0], hand_[0][1]-hand_[18][1]),
            (hand_[19][0]-hand_[20][0], hand_[19][1]-hand_[20][1])
        ))
        return angle_list

    @staticmethod
    def _hand_pos(finger_angle):
        f1, f2, f3, f4, f5 = finger_angle
        # Simplified gesture mapping
        if f1 >= 50 and f2 < 50 and f3 >= 50 and f4 >= 50 and f5 >= 50: return '1'
        elif f1 >= 50 and f2 < 50 and f3 < 50 and f4 >= 50 and f5 >= 50: return '2'
        elif f1 >= 50 and f2 < 50 and f3 < 50 and f4 < 50 and f5 > 50: return '3'
        elif f1 >= 50 and f2 < 50 and f3 < 50 and f4 < 50 and f5 < 50: return '4'
        elif f1 < 50 and f2 < 50 and f3 < 50 and f4 < 50 and f5 < 50: return '5' # Palm open for STOP or selecting joint 5
        elif f1 < 50 and f2 >= 50 and f3 >= 50 and f4 >= 50 and f5 < 50: return '6'
        
        # Left hand gestures for control
        # Condition 1 for '0' (e.g., thumb up, others bent)
        elif f1 < 50 and f2 >= 50 and f3 >= 50 and f4 >= 50 and f5 >= 50: return '0' 
        # Condition 2 for '0' (e.g., fist - all fingers bent) - ADDED
        elif f1 >= 50 and f2 >= 50 and f3 >= 50 and f4 >= 50 and f5 >= 50: return '0'
        
        else: return ''

    def _handle_left_hand_relative_continuous(self, gesture_str, current_mcp_y, current_mcp_x):
        if gesture_str == '0': # Fist for engage/control
            if not self.left_control_active:
                self.left_control_active = True
                self.left_initial_y = current_mcp_y
                self.left_initial_x = current_mcp_x
                self.action_internal = "engage_control"
                # Ensure self.selected_joint_internal is handled if None for f-string
                self.debug_text = f"L:Engaged J{self.selected_joint_internal if self.selected_joint_internal is not None else 'N/A'}"
                self.debug_text_color = (0,255,0)
            else: 
                delta_y = current_mcp_y - self.left_initial_y
                if abs(delta_y) > self.sensitivity_threshold:
                    if delta_y > 0: 
                        self.action_internal = "set_angle_continuous"
                        self.value_internal = 1.0 
                        self.debug_text = f"L:Controlling J{self.selected_joint_internal if self.selected_joint_internal is not None else 'N/A'} (Down)"
                    else: 
                        self.action_internal = "set_angle_continuous"
                        self.value_internal = -1.0 
                        self.debug_text = f"L:Controlling J{self.selected_joint_internal if self.selected_joint_internal is not None else 'N/A'} (Up)"
                    self.debug_text_color = (0,255,255)
                else: 
                    self.action_internal = "hold"
                    self.value_internal = 0.0
                    self.debug_text = f"L:Hold J{self.selected_joint_internal if self.selected_joint_internal is not None else 'N/A'}"
                    self.debug_text_color = (255,150,0)
        elif gesture_str == '5': 
            if self.left_control_active:
                self.left_control_active = False
                self.action_internal = "disengage_control"
                self.value_internal = 0.0
                self.debug_text = f"L:Disengaged J{self.selected_joint_internal if self.selected_joint_internal is not None else 'N/A'}"
                self.debug_text_color = (0,0,255)

    def _handle_left_hand_absolute(self, gesture_str, current_mcp_y):
        if gesture_str == '0': 
            if not self.left_control_active:
                self.left_control_active = True
                self.action_internal = "engage_absolute_control"
                self.debug_text = f"L:Abs Engaged J{self.selected_joint_internal if self.selected_joint_internal is not None else 'N/A'}"
                self.debug_text_color = (0,255,0)

            if self.left_control_active: 
                top_dead_zone_end_y = self.h * self.absolute_control_dead_zone_v_ratio
                bottom_dead_zone_start_y = self.h * (1 - self.absolute_control_dead_zone_v_ratio)
                active_zone_height = bottom_dead_zone_start_y - top_dead_zone_end_y

                if active_zone_height <= 0: 
                    self.action_internal = "hold" 
                    self.value_internal = 0.0
                    self.debug_text = "L:Abs Dead Zone Error"
                    self.debug_text_color = (255,0,0)
                else:
                    y_pos_in_frame = current_mcp_y
                    
                    if y_pos_in_frame < top_dead_zone_end_y: 
                        normalized_y = 0.0 
                        self.action_internal = "set_angle_absolute"
                        self.value_internal = normalized_y
                        self.debug_text = f"L:Abs J{self.selected_joint_internal if self.selected_joint_internal is not None else 'N/A'} (Max)"
                        self.debug_text_color = (0,255,255)
                    elif y_pos_in_frame > bottom_dead_zone_start_y: 
                        normalized_y = 1.0 
                        self.action_internal = "set_angle_absolute"
                        self.value_internal = normalized_y
                        self.debug_text = f"L:Abs J{self.selected_joint_internal if self.selected_joint_internal is not None else 'N/A'} (Min)"
                        self.debug_text_color = (0,255,255)
                    else: 
                        normalized_y = (y_pos_in_frame - top_dead_zone_end_y) / active_zone_height
                        normalized_y = max(0.0, min(1.0, normalized_y)) 

                        self.action_internal = "set_angle_absolute"
                        self.value_internal = normalized_y 
                        self.debug_text = f"L:Abs J{self.selected_joint_internal if self.selected_joint_internal is not None else 'N/A'} ({normalized_y:.2f})"
                        self.debug_text_color = (0,255,255)
        
        elif gesture_str == '5': 
            if self.left_control_active:
                self.left_control_active = False
                self.action_internal = "disengage_absolute_control"
                self.value_internal = 0.0 
                self.debug_text = f"L:Abs Disengaged J{self.selected_joint_internal if self.selected_joint_internal is not None else 'N/A'}"
                self.debug_text_color = (0,0,255)

    def _handle_left_hand_discrete(self, gesture_str):
        if gesture_str == '0': 
            self.action_internal = "increase"
            self.debug_text = f"L:Increase J{self.selected_joint_internal if self.selected_joint_internal is not None else 'N/A'}"
            self.debug_text_color = (0,255,0)
        elif gesture_str == '6': 
            self.action_internal = "decrease"
            self.debug_text = f"L:Decrease J{self.selected_joint_internal if self.selected_joint_internal is not None else 'N/A'}"
            self.debug_text_color = (0,255,0)
        elif gesture_str == '5': 
            self.action_internal = "stop_discrete" 
            self.debug_text = f"L:Stop J{self.selected_joint_internal if self.selected_joint_internal is not None else 'N/A'}"
            self.debug_text_color = (0,0,255)

    def get_command(self, frame):
        # Flip the frame horizontally for a mirror effect
        flipped_frame = cv2.flip(frame, 1)

        # --- MODIFICATION START: Resize frame and prepare for MediaPipe ---
        # Resize to consistent dimensions (self.w, self.h)
        # This 'image' will be BGR and is used for all drawing and final output
        image = cv2.resize(flipped_frame, (self.w, self.h))
        
        # Convert a copy to RGB for MediaPipe processing
        image_rgb_for_mediapipe = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb_for_mediapipe.flags.writeable = False
        results = self.hands.process(image_rgb_for_mediapipe)
        image_rgb_for_mediapipe.flags.writeable = True # Good practice
        # --- MODIFICATION END ---

        self.current_left_wrist_y = None # Reset current positions
        self.current_left_wrist_x = None
        self.current_right_wrist_y = None # For potential future use or consistency
        self.current_right_wrist_x = None

        current_right_gesture = None
        current_left_gesture = None
        
        self.action_internal = None
        self.value_internal = 0.0

        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # --- MODIFICATION START: Draw hand landmarks ---
                self.mp_drawing.draw_landmarks(
                    image, # Draw on the BGR 'image'
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
                # --- MODIFICATION END ---
                
                handedness = results.multi_handedness[i]
                lm_list = []
                for lm in hand_landmarks.landmark:
                    lm_list.append([lm.x, lm.y, lm.z]) 

                finger_angle = self._hand_angle(lm_list)
                gesture_str = self._hand_pos(finger_angle)

                if handedness.classification[0].label == 'Left':
                    current_left_gesture = gesture_str
                    mcp_landmark = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP]
                    self.current_left_wrist_y = mcp_landmark.y * self.h 
                    self.current_left_wrist_x = mcp_landmark.x * self.w

                    if self.control_mode == "relative_continuous":
                        self._handle_left_hand_relative_continuous(current_left_gesture, self.current_left_wrist_y, self.current_left_wrist_x)
                    elif self.control_mode == "absolute":
                        self._handle_left_hand_absolute(current_left_gesture, self.current_left_wrist_y)
                    elif self.control_mode == "discrete":
                        self._handle_left_hand_discrete(current_left_gesture)

                elif handedness.classification[0].label == 'Right':
                    current_right_gesture = gesture_str
                    # --- MODIFICATION START: Use self.h/w for coordinate scaling ---
                    wrist_landmark = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]
                    self.current_right_wrist_y = wrist_landmark.y * self.h 
                    self.current_right_wrist_x = wrist_landmark.x * self.w
                    # --- MODIFICATION END ---

                    if '1' <= gesture_str <= '6':
                        newly_selected_joint = int(gesture_str)
                        if self.selected_joint_internal != newly_selected_joint:
                            self.selected_joint_internal = newly_selected_joint
                            self.debug_text = f"R:Selected Joint {self.selected_joint_internal}"
                            self.debug_text_color = (255,255,0) # Yellow for selection
                            # Reset left hand control state upon new joint selection
                            self.left_control_active = False 
            
            if self.selected_joint_internal is not None:
                pass 
            
            else: 
                self.debug_text = "R:Select Joint (1-6)"
                self.debug_text_color = (255, 0, 255) # Magenta to prompt selection
                self.action_internal = None
                self.value_internal = 0.0
                if self.left_control_active: # If a joint was unselected while L-control was active
                    self.left_control_active = False
        
        else: # No hands detected
            # ...existing code...
            if self.left_control_active: print("DEBUG: L-Ctrl DISENGAGED (No hands detected)")
            self.left_control_active = False
            self.left_initial_y = None
            # self.action_internal = None # Or "stop" if preferred when hands disappear
            # self.value_internal = 0.0
            # self.selected_joint_internal = None # Keep selected joint or clear?
            self.debug_text = "No Hands Detected"
            self.debug_text_color = (0, 0, 255)

        # Draw initial Y line and current Y dot if control is active (for relative mode)
        if self.control_mode == "relative_continuous" and self.left_control_active and self.left_initial_y is not None and self.current_left_wrist_y is not None:
            if self.current_left_wrist_x is not None:                
                cv2.line(image, (int(self.left_initial_x) - 50, int(self.left_initial_y)), 
                         (int(self.left_initial_x) + 50, int(self.left_initial_y)), (255, 0, 0), 2) # Blue line for initial Y
                cv2.circle(image, (int(self.current_left_wrist_x), int(self.current_left_wrist_y)), 
                           5, (0, 0, 255), -1) # Red dot for current wrist Y
        # elif self.control_mode == "absolute" and self.left_control_active and self.current_left_wrist_x is not None and self.current_left_wrist_y is not None:
            # print(f"DEBUG_DRAW_ORANGE_DOT: X={self.current_left_wrist_x:.2f}, Y={self.current_left_wrist_y:.2f}, W={self.w}, H={self.h}")
            # # For absolute mode, just draw the current control point
            # cv2.circle(image, (int(self.current_left_wrist_x), int(self.current_left_wrist_y)), 
            #            7, (0, 165, 255), -1) # Orange dot for absolute control point
            
            # # Optionally, draw the dead zones for absolute control
            # top_dead_zone_px = int(self.h * self.absolute_control_dead_zone_v_ratio)
            # bottom_dead_zone_px = int(self.h * (1 - self.absolute_control_dead_zone_v_ratio))
            # cv2.line(image, (0, top_dead_zone_px), (self.w, top_dead_zone_px), (100, 100, 100), 1)
            # cv2.line(image, (0, bottom_dead_zone_px), (self.w, bottom_dead_zone_px), (100, 100, 100), 1)


        # Display debug text and selected joint/action
        cv2.putText(image, self.debug_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.debug_text_color, 2)
        
        # Display general instructions
        instruction_text = "R:1-6 Select Joint | L-Fist:Control | L-Palm:Stop"
        cv2.putText(image, instruction_text, (10, self.h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        
        # Return the recognized command, the value (if any), and the annotated image
        return self.selected_joint_internal, self.action_internal, self.value_internal, image

    def release(self):
        self.hands.close()

# Example usage (for testing this module directly)
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    # gesture_recognizer = GestureRecognizer(control_mode="relative_continuous", sensitivity_threshold=20, angle_step_continuous=0.4)
    gesture_recognizer = GestureRecognizer(control_mode="absolute", absolute_control_dead_zone_v_ratio=0.15)


    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    cv2.namedWindow('Gesture Control', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Gesture Control', 960, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = cv2.flip(frame, 1)
        
        joint, action, value, annotated_image = gesture_recognizer.get_command(frame)

        if joint is not None and action is not None:
            if action == "set_angle_continuous":
                 print(f"Joint: {joint}, Action: {action}, Value: {value:.3f}")
            elif action == "set_angle_absolute":
                 print(f"Joint: {joint}, Action: {action}, Normalized Value: {value:.3f}")
            else:
                 print(f"Joint: {joint}, Action: {action}")


        cv2.imshow('Gesture Control', annotated_image)

        if cv2.waitKey(5) & 0xFF == 27:  # ESC key to break
            break

    gesture_recognizer.release()
    cap.release()
    cv2.destroyAllWindows()
