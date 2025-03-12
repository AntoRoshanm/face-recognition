from flask import Flask, render_template, Response, request, jsonify, redirect
import cv2
import os
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
import base64

app = Flask(__name__)
data_folder = "store_dataset"
model_path = "face_model.pkl"
os.makedirs(data_folder, exist_ok=True)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Function to extract faces
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    return faces

# Train the model
def train_model():
    faces = []
    labels = []
    userlist = os.listdir(data_folder)
    for user in userlist:
        user_path = os.path.join(data_folder, user)
        if os.path.isdir(user_path):
            for imgname in os.listdir(user_path):
                img = cv2.imread(os.path.join(user_path, imgname))
                resized_face = cv2.resize(img, (100, 100)).flatten()
                faces.append(resized_face)
                labels.append(user)
    if faces:
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(np.array(faces), labels)
        joblib.dump(knn, model_path)

# Load the trained model
def load_model():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# Route to serve the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to store dataset
@app.route('/store_dataset', methods=['GET', 'POST'])
def store_dataset():
    if request.method == 'POST':
        name = request.form['name']
        person_folder = os.path.join(data_folder, name)
        os.makedirs(person_folder, exist_ok=True)
        cap = cv2.VideoCapture(0)
        count = 0
        while count < 100:
            ret, frame = cap.read()
            if not ret:
                break
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (100, 100))
                cv2.imwrite(f'{person_folder}/{count}.jpg', face_resized)
                count += 1
            cv2.imshow('Capturing Faces', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        train_model()  # Train model after dataset update
        return redirect('/')
    return render_template('store_dataset.html')

# Route to render the face recognition page
@app.route('/face_recognition')
def face_recognition():
    return render_template('face_recognition.html')

# Route to process frames for face recognition
@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.json
    image_data = data['image'].split(',')[1]  # Remove the data URL prefix
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform face detection and recognition
    faces = extract_faces(frame)
    model = load_model()
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (100, 100)).flatten().reshape(1, -1)
        if model:
            identity = model.predict(face_resized)[0]
            cv2.putText(frame, identity, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Encode the processed frame back to base64
    _, buffer = cv2.imencode('.jpg', frame)
    processed_image = base64.b64encode(buffer).decode('utf-8')
    return jsonify({'image': f'data:image/jpeg;base64,{processed_image}'})

if __name__ == '__main__':
    app.run(debug=True)
