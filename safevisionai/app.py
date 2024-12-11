import logging
import os
from flask import Flask, render_template, Response, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_mail import Mail, Message
import cv2
import torch
import threading
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from torch import nn
from PIL import Image
from threading import Lock

# Initialize Flask app
app = Flask(__name__)
# --- Logging Configuration ---
LOG_DIR = os.path.join(os.getcwd(), "logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "app.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    filemode="a",
)
logger = logging.getLogger(__name__)



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

# Camera Initialization
def initialize_camera():
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                logger.error("Unable to open camera.")
                return None
            logger.info("Camera initialized successfully.")
        return camera

# Release Camera
def release_camera():
    global camera
    with camera_lock:
        if camera and camera.isOpened():
            camera.release()
            camera = None
            logger.info("Camera released.")


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
    logger.info("Index page accessed.")
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
    global streaming
    if not streaming:
        logger.warning("Stream requested but camera is not active.")
        return Response("Camera not active.", status=503)
    logger.info("Video stream started.")
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recording/<int:camera_id>')
@login_required
def start_recording(camera_id):
    global streaming, recording
    if not initialize_camera():
        flash("Unable to start camera.")
        logger.error("Failed to initialize camera for recording.")
        return redirect(url_for('dashboard'))
    streaming = True
    recording = True
    thread = threading.Thread(target=detect_weapon, args=(camera_id, session.get('email')))
    thread.start()
    logger.info("Recording started.")
    flash("Recording started.")
    return redirect(url_for('dashboard'))

@app.route('/stop_recording')
@login_required
def stop_recording():
    global streaming, recording
    streaming = False
    recording = False
    release_camera()
    logger.info("Recording stopped.")
    flash("Recording stopped.")
    return redirect(url_for('dashboard'))



def detect_weapon(camera_id, email):
    """Detect weapons and violence using YOLO and send email alerts."""
    global recording, camera

    with app.app_context():  # Ensure Flask app context is available for the thread
        camera = initialize_camera()  # Get the camera object
        if camera is None:
            logger.error("Camera initialization failed in weapon detection.")
            return

        logger.info("Camera opened successfully. Starting detection...")

        while recording:
            success, frame = camera.read()
            if not success:
                logger.error("Failed to grab frame.")
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
                    logger.info("Threat detected")
                    weapon_detected = True
                    image_path = './detections/threat_detected.jpg'
                    if not os.path.exists('detections'):
                        os.makedirs('detections')
                    cv2.imwrite(image_path, frame)
                    send_email(image_path, email)
                    logger.info("Threat detected and email sent!")
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
            logger.info("Raw outputs: {outputs}")
            logger.info("Predicted: {violence_classes[predicted.item()]}, Confidence: {confidence.item():.2f}")

            # Check if confidence exceeds threshold
            if confidence.item() >= 1 and violence_classes[predicted.item()] == "Violence":
                logger.info(f"Violence detected with confidence: {confidence.item():.2f}")
                image_path = './detections/violence_detected.jpg'
                cv2.imwrite(image_path, frame)
                send_email(image_path, email)
                logger.info("Violence threat detected and email sent!")
            else:
                logger.info("No violence detected. Confidence: {confidence.item():.2f}")

        # Cleanup when recording stops
        release_camera()
        logger.info("Camera released.")


# Email notification
def send_email(image_path, email):
    msg = Message("Threat Detected!", sender="christi003@gmail.com", recipients=[email])
    msg.body = "A threat was detected. Check the attached image."
    with app.open_resource(image_path) as fp:
        msg.attach(image_path, "image/jpeg", fp.read())
    mail.send(msg)
    logger.info("Email sent successfully.")

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    logger.info("Starting the Flask app.")
    app.run(debug=True, port=5112)
