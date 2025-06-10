from flask import Flask, render_template, request, send_from_directory
import os
from car_detection_hmv_lmv import run_vehicle_detection

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    video = request.files['video']
    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)

    # Run detection and tracking
    run_vehicle_detection(video_path)

    return render_template('result.html',
                           video_file='static/output_video.mp4',
                           csv_file='static/counts.csv')

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
