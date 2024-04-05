from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    while True:
        # Replace '0' with the URL of your IP webcam stream
        cap = cv2.VideoCapture("http://192.168.0.190:8080/video")
        success, frame = cap.read()
        if not success:
            break
        else:
            # Here you can implement your neural network model to process 'frame'
            # Example: processed_frame = your_neural_network_model(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
