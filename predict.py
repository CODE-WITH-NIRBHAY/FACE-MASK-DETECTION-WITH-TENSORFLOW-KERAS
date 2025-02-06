import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('face_mask_model.h5')

def predict_mask():
    # Set up face detection (Haar Cascade Classifier)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)  # Open webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        frame = cv2.flip(frame, 1)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)  # Detect faces

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw rectangle around face

            # Extract the face from the frame and resize it
            face_img = frame[y:y+h, x:x+w]
            face_img_resized = cv2.resize(face_img, (150, 150))  # Resize for model input
            face_img_array = image.img_to_array(face_img_resized) / 255.0  # Normalize the image
            face_img_array = np.expand_dims(face_img_array, axis=0)  # Add batch dimension

            # Predict if the face has a mask
            prediction = model.predict(face_img_array)
            if prediction[0] > 0.5:
                label = "With Mask"
                color = (0, 255, 0)  # Green for with mask
            else:
                label = "Without Mask"
                color = (0, 0, 255)  # Red for without mask

            # Display the label and prediction on the frame
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Show the webcam frame
        cv2.imshow('Mask Detection', frame)

        # Break the loop when 'n' is pressed
        if cv2.waitKey(1) & 0xFF == ord('n'):
            break

    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close all OpenCV windows

if __name__ == "__main__":
    predict_mask()  # Start the mask prediction process
