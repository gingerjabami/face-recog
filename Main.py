from datetime import datetime, date
import streamlit as st
import cv2
import numpy as np
import face_recognition
import os
import pandas as pd
from PIL import Image

# Configuration and Styling
st.set_page_config(page_title="MP Police Face Recognition System", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f7f9fc;
        }
        h1 {
            text-align: center;
            color: #0047AB;
            font-weight: bold;
        }
        .sidebar .sidebar-content {
            background-color: #1E2F5C;
            color: white;
        }
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            background-color: #0047AB;
            color: white;
            text-align: center;
            padding: 10px;
            font-size: 14px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>Madhya Pradesh Police Attendance System</h1>", unsafe_allow_html=True)

# Sidebar: Configuration
st.sidebar.header("System Configuration")
current_date = st.sidebar.date_input("Date", datetime.today())
current_time = st.sidebar.time_input("Time", datetime.now().time())
stop_button = st.sidebar.button("Stop Attendance System")

# Directory for storing images
IMAGE_DIR = "FaceAtten_Excel/ImagesAt"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Attendance CSV
ATTENDANCE_FILE = "Attendance.csv"

# Handle Uploaded Images
uploaded_image = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
if uploaded_image:
    image_path = os.path.join(IMAGE_DIR, uploaded_image.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())
    st.sidebar.success("Image uploaded successfully!")

# Load Images and Encode Faces
def load_images_and_encode(directory):
    images, classNames = [], []
    for file_name in os.listdir(directory):
        img_path = os.path.join(directory, file_name)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            classNames.append(os.path.splitext(file_name)[0])
    return images, classNames

@st.cache_resource
def find_encodings(images):
    encodings = []
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encoding = face_recognition.face_encodings(img_rgb)[0]
            encodings.append(encoding)
        except IndexError:
            pass  # Skip images without detectable faces
    return encodings

images, classNames = load_images_and_encode(IMAGE_DIR)
encodeListKnown = find_encodings(images)
st.sidebar.success(f"Loaded {len(encodeListKnown)} face encodings.")

# Load or Create Attendance Log
def load_attendance():
    # Check if the file exists
    if not os.path.exists(ATTENDANCE_FILE):
        print(f"Attendance file not found. Creating a new file at {ATTENDANCE_FILE}.")
        empty_df = pd.DataFrame(columns=["Name", "Date", "Time"])
        empty_df.to_csv(ATTENDANCE_FILE, index=False)
        return empty_df

    # Handle empty file scenario
    try:
        df = pd.read_csv(ATTENDANCE_FILE)
        if df.empty:
            print(f"Attendance file {ATTENDANCE_FILE} is empty. Reinitializing with headers.")
            df = pd.DataFrame(columns=["Name", "Date", "Time"])
            df.to_csv(ATTENDANCE_FILE, index=False)
        return df
    except pd.errors.EmptyDataError:
        print(f"Attendance file {ATTENDANCE_FILE} is corrupted or empty. Reinitializing.")
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
        df.to_csv(ATTENDANCE_FILE, index=False)
        return df
def save_attendance(name):
    now = datetime.now()
    attendance_df = load_attendance()
    new_entry = {"Name": name, "Date": now.strftime("%B %d, %Y"), "Time": now.strftime("%H:%M:%S")}
    attendance_df = pd.concat([attendance_df, pd.DataFrame([new_entry])], ignore_index=True)
    attendance_df.drop_duplicates(subset=["Name", "Date"], keep="last", inplace=True)  # Avoid duplicate entries
    attendance_df.to_csv(ATTENDANCE_FILE, index=False)

attendance_df = load_attendance()
st.sidebar.markdown("### Attendance History")
st.sidebar.dataframe(attendance_df)

# Real-Time Camera Feed and Attendance
st.markdown("<h2 style='text-align: center;'>Real-Time Attendance System</h2>", unsafe_allow_html=True)
frame_placeholder = st.empty()
marked_names = set(attendance_df["Name"].tolist())

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.warning("Unable to access the camera. Please check your device settings.")

while cap.isOpened() and not stop_button:
    ret, img = cap.read()
    if not ret:
        st.warning("Unable to capture frames.")
        break

    imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(imgS)
    encodings = face_recognition.face_encodings(imgS, face_locations)

    for encoding, face_location in zip(encodings, face_locations):
        matches = face_recognition.compare_faces(encodeListKnown, encoding)
        face_distances = face_recognition.face_distance(encodeListKnown, encoding)
        match_index = np.argmin(face_distances) if matches else None

        if match_index is not None and matches[match_index]:
            name = classNames[match_index]
            if name not in marked_names:
                save_attendance(name)
                marked_names.add(name)

            y1, x2, y2, x1 = [v * 4 for v in face_location]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(img_rgb, channels="RGB")

    if stop_button:
        break

cap.release()
cv2.destroyAllWindows()

# Footer
st.markdown("""
    <div class='footer'>
        © 2024 Madhya Pradesh Police | Powered by Aditya Bhattacharya
    </div>
""", unsafe_allow_html=True)
