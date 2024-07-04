import cv2
import mediapipe as mp

# Initializing mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Load images for gestures
gesture_images = {
    "Open Hand": cv2.imread('imp_18.png'),
    "Pointing Up": cv2.imread('img_27.png'),
    "Default": cv2.imread('brah.png')
}

# Function to overlay an image on the frame
def overlay_image(frame, img, x, y, scale=1):
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    h, w, _ = img.shape
    frame[y:y+h, x:x+w] = img

# Initializing Video Capture
cap = cv2.VideoCapture(0)

# Capturing and processing each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmark_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_list.append([cx, cy])
            if len(landmark_list) != 0:
                gesture_image = None
                # Example logic for gesture recognition
                # Open Hand (Palm) Gesture
                if landmark_list[4][1] < landmark_list[3][1] and landmark_list[8][1] < landmark_list[6][1]:
                    gesture_image = gesture_images["Open Hand"]
                # Pointing Up Gesture
                elif landmark_list[4][1] > landmark_list[3][1] and landmark_list[8][1] < landmark_list[6][1]:
                    gesture_image = gesture_images["Pointing Up"]
                else:
                    gesture_image = gesture_images["Default"]
                
                # Display the corresponding image
                if gesture_image is not None:
                    overlay_image(frame, gesture_image, landmark_list[0][0] - 50, landmark_list[0][1] - 150, scale=0.5)
    
    # Display the Frame
    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()