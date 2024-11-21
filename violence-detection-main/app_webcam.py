import cv2
import numpy as np
import streamlit as st
from PIL import Image


@st.cache(allow_output_mutation=True)
def get_predictor_model():
    from model import Model
    model = Model()
    return model


# Streamlit app interface
header = st.container()
model = get_predictor_model()

with header:
    st.title('Hello!')
    st.text(
        'Using this app, you can classify whether there is a fight on a street, fire, car crash, or everything is okay!'
    )

# Add a button to start the webcam feed
start_webcam = st.button("Start Webcam")

if start_webcam:
    st.write("Webcam started. Please allow access to your webcam.")

    # Open webcam feed
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Could not access the webcam.")
    else:
        frame_placeholder = st.empty()  # Placeholder for displaying the webcam frame
        prediction_placeholder = st.empty()  # Placeholder for showing predictions

        # Read frames in a loop
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Could not read frame from webcam.")
                break

            # Convert the frame to RGB for PIL and prediction
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)

            # Perform prediction
            prediction = model.predict(image=np.array(image))
            label_text = prediction['label'].title()
            confidence = prediction['confidence']

            # Display prediction
            prediction_placeholder.markdown(
                f"### Predicted Label: **{label_text}**\nConfidence: **{confidence:.2f}**"
            )

            # Display the webcam feed
            frame_placeholder.image(rgb_frame, channels="RGB")

            # Stop webcam feed when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

st.write("Webcam feed stopped.")
