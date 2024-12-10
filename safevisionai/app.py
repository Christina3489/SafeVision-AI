from flask import Flask, render_template, Response, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_mail import Mail, Message
import cv2
import torch
import os
import threading
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from torch import nn
from PIL import Image
from threading import Lock

# Initialize Flask app
app = Flask(__name__)

# Global variables
camera = None
camera_lock = Lock()
streaming = False
recording = False

# Configure SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your_secret_key'

# Initialize database
db = SQLAlchemy(app)

# Email configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'christinacad04@gmail.com'
app.config['MAIL_PASSWORD'] = 'kocl ptoo dxsq nvdx'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)

# Load YOLO model for weapon detection
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/chris/Downloads/weapon_detection/weapon_detection/weapon_detection_yolo/yolov5/runs/train/exp/weights/best.pt', force_reload=True)

# Load Violence Detection Model
violence_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
violence_model.fc = nn.Linear(violence_model.fc.in_features, 2)
violence_model.load_state_dict(torch.load("C:/Users/chris/Downloads/weapon_detection/weapon_detection/best_violence_detection_model.pth"))
violence_model.eval()
violence_classes = ["Non-Violence", "Violence"]

# Flask-Login Initialization
login_manager = LoginManager()
login_manager.init_app(app)

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    email = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)

# Camera model
class Camera(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    room_name = db.Column(db.String(150), nullable=False)
    camera_url = db.Column(db.String(150), nullable=False)

# Login manager user loader
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def initialize_camera():
    """Initialize the camera for video capture."""
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)  # Use default camera
            if not camera.isOpened():
                print("Error: Unable to open camera.")
                return None  # Return None instead of False
        return camera  # Return the camera object

# Release camera
def release_camera():
    """Release the camera when not in use."""
    global camera
    with camera_lock:
        if camera and camera.isOpened():
            camera.release()
            camera = None
            print("Camera released.")

# Generate video frames for streaming
def generate_frames():
    """Generate frames for video feed."""
    global camera
    while streaming:
        with camera_lock:
            if not camera or not camera.isOpened():
                break
            success, frame = camera.read()
            if not success:
                print("Error: Failed to grab frame.")
                break
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Routes
@app.route('/')
def index():
    return render_template('index.html', title="SafeVision-AI")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html', title="Register | CarForAnyone")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            session['email'] = email
            return redirect(url_for('dashboard'))
        else:
            flash('Login failed. Check your email and password.')
    return render_template('login.html', title="Login | CarForAnyone")

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', title="Dashboard | SafeVision-AI")

@app.route('/video_feed')
def video_feed():
    """Route to stream video feed to HTML."""
    global streaming
    if not streaming:
        print("Stream requested but camera is not active.")
        return Response("Camera not active.", status=503)
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recording/<int:camera_id>')
@login_required
def start_recording(camera_id):
    global streaming, recording
    if not initialize_camera():
        flash("Unable to start camera.")
        return redirect(url_for('dashboard'))
    streaming = True
    recording = True
    thread = threading.Thread(target=detect_weapon, args=(camera_id, session.get('email')))
    thread.start()
    flash("Recording started.")
    return redirect(url_for('dashboard'))

@app.route('/stop_recording')
@login_required
def stop_recording():
    global streaming, recording
    streaming = False
    recording = False
    release_camera()
    flash("Recording stopped.")
    return redirect(url_for('dashboard'))


def detect_weapon(camera_id, email):
    """Detect weapons and violence using YOLO and send email alerts."""
    global recording, camera

    with app.app_context():  # Ensure Flask app context is available for the thread
        camera = initialize_camera()  # Get the camera object
        if camera is None:
            print("Camera initialization failed.")
            return

        print("Camera opened successfully. Starting detection...")

        while recording:
            success, frame = camera.read()
            if not success:
                print("Error: Failed to grab frame.")
                continue  # Keep trying to read the next frame

            print("Frame captured. Running YOLO detection...")
            
            # Initialize detection flags
            weapon_detected = False
            violence_detected = False

            # YOLO Weapon Detection
            results = model(frame)
            for result in results.xyxy[0]:
                class_name = model.names[int(result[5])]
                if class_name == 'Handgun' or class_name == 'knife'or class_name == 'Fight' or class_name == 'Fire':
                    print(f"Threat detected")
                    weapon_detected = True
                    image_path = './detections/threat_detected.jpg'
                    if not os.path.exists('detections'):
                        os.makedirs('detections')
                    cv2.imwrite(image_path, frame)
                    send_email(image_path, email)
                    print("Threat detected and email sent!")
                    break  # Exit the current detection loop, but continue overall recording loop

            # Violence Detection with Threshold
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Apply transformations
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            input_tensor = transform(pil_image).unsqueeze(0)

            # Model inference
            violence_model.eval()
            outputs = torch.nn.functional.softmax(violence_model(input_tensor), dim=1)
            confidence, predicted = torch.max(outputs, 1)

            # Log outputs for debugging
            print(f"Raw outputs: {outputs}")
            print(f"Predicted: {violence_classes[predicted.item()]}, Confidence: {confidence.item():.2f}")

            # Check if confidence exceeds threshold
            if confidence.item() >= 1 and violence_classes[predicted.item()] == "Violence":
                print(f"Violence detected with confidence: {confidence.item():.2f}")
                image_path = './detections/violence_detected.jpg'
                cv2.imwrite(image_path, frame)
                send_email(image_path, email)
                print("Violence threat detected and email sent!")
            else:
                print(f"No violence detected. Confidence: {confidence.item():.2f}")

        # Cleanup when recording stops
        release_camera()
        print("Camera released.")


# Email notification
def send_email(image_path, email):
    msg = Message("Threat Detected!", sender="christi003@gmail.com", recipients=[email])
    msg.body = "A threat was detected. Check the attached image."
    with app.open_resource(image_path) as fp:
        msg.attach(image_path, "image/jpeg", fp.read())
    mail.send(msg)
    print("Email sent successfully.")

# Run the app
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5112)
