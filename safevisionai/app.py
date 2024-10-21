from flask import Flask, render_template, request, redirect, session, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_mail import Mail, Message
import cv2
import torch
import os
import threading

# Initialize Flask app
app = Flask(__name__)

# Configure SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your_secret_key'

# Initialize database
db = SQLAlchemy(app)

# Email configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'pcprasant376@gmail.com'
app.config['MAIL_PASSWORD'] = 'eeci vthn spos aqwr'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='E:/Projects/weapon_detection/weapon_detection_yolo/yolov5/runs/train/exp/weights/best.pt', force_reload=True)



# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)

# Global variable to control the recording
recording = False
thread_running = False

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

# Initialize the database
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Home page route (index.html)
@app.route('/')
def index():
    return render_template('index.html')

# User registration route
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
    return render_template('register.html')

# User login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            session['email']=email
            print(session['email'])
            return redirect(url_for('dashboard'))
        else:
            flash('Login failed. Check your email and password.')
    return render_template('login.html')

# User logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# Dashboard route
@app.route('/dashboard')
@login_required
def dashboard():
    user_cameras = Camera.query.filter_by(user_id=current_user.id).all()
    return render_template('dashboard.html', name=current_user.username, cameras=user_cameras)

# Camera registration route
@app.route('/add_camera', methods=['GET', 'POST'])
@login_required
def add_camera():
    if request.method == 'POST':
        room_name = request.form['room_name']
        camera_url = request.form['camera_url']
        new_camera = Camera(user_id=current_user.id, room_name=room_name, camera_url=camera_url)
        db.session.add(new_camera)
        db.session.commit()
        return redirect(url_for('dashboard'))
    return render_template('add_camera.html')

# Route to start the recording process
@app.route('/start_recording/<int:camera_id>')
@login_required
def start_recording(camera_id):
    global recording
    global thread_running

    if not thread_running:
        recording = True  # Set global flag to start recording
        thread = threading.Thread(target=detect_weapon, args=(camera_id,session['email']))
        thread.start()
        thread_running = True  # Indicate thread has started
        print(f"Recording started for camera ID: {camera_id}")
    else:
        print("Recording already in progress...")

    return redirect(url_for('dashboard'))

# Route to stop recording
@app.route('/stop_recording')
@login_required
def stop_recording():
    global recording
    global thread_running
    recording = False  # Set global flag to stop recording
    thread_running = False
    print("Recording stopped.")
    return redirect(url_for('dashboard'))

def detect_weapon(camera_id,email):
    global recording
    global thread_running

    # Ensure that the app context is available for database queries in the thread
    with app.app_context():
        camera = Camera.query.get_or_404(camera_id)

        # Use laptop webcam if camera URL is '0'
        camera_url = 0 if camera.camera_url == '0' else camera.camera_url
        print(f"Accessing camera: {camera_url}")

        cap = cv2.VideoCapture(camera_url)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_url}.")
            thread_running = False
            return

        print("Camera opened successfully. Starting detection...")

        while recording:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame.")
                break

            print("Frame captured. Running YOLO detection...")

            # Show the captured frame in a window
            cv2.imshow('Camera Feed', frame)

            # Break the loop if 'q' is pressed or the recording is stopped
            if cv2.waitKey(1) & 0xFF == ord('q') or not recording:
                print("Stopping recording.")
                break

            # YOLOv5 detectionq
            results = model(frame)
            print(',,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,')
            for result in results.xyxy[0]:
                class_name = model.names[int(result[5])]
                print(class_name,'////////////////////////////////////////////////')
                if class_name == 'Handgun' or class_name == 'knife':
                    print('hello11111111111',current_user)
                    image_path = f'./detections/room.jpg'
                    if not os.path.exists('detections'):
                        os.makedirs('detections')
                    cv2.imwrite(image_path, frame)
                    send_email(image_path,email)
                    print("Weapon detected and email sent!")
                    break

        # Release camera and close window
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released.")
        thread_running = False



# Function to send an email notification
def send_email(image_path, email):
    msg = Message('Weapon Detected!', sender='pcprasant376@gmail.com', recipients=[email])
    msg.body = 'A weapon was detected in your home camera. Please check the attached image.'
    with app.open_resource(image_path) as fp:
        msg.attach(image_path, 'image/jpeg', fp.read())
    mail.send(msg)

# Delete camera route
@app.route('/delete_camera/<int:camera_id>')
@login_required
def delete_camera(camera_id):
    camera = Camera.query.get_or_404(camera_id)
    db.session.delete(camera)
    db.session.commit()
    print(f"Camera {camera.room_name} deleted.")
    return redirect(url_for('dashboard'))

# Run the Flask app and create the database tables if not already created
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True,port=5112)
