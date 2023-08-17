#Notes
#Class 1 = Male
#Class 0 = Female
#add credit kid
from flask import *
import cv2
import torch
import math

app = Flask(__name__)

camera = cv2.VideoCapture(0) 

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
camModel = torch.hub.load('ultralytics/yolov5', 'custom', path='flaskLiveWebcamIntegration/bestModel.pt', force_reload=True)

faceLimit = 1
faceCount = 0
#video stream frames
def gen_frames():
    global faceLimit
    global faceCount  
    while True:
        
        success, frame = camera.read()  
        if not success:
            break
        else:
            faceCount = 0
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = camModel(frameRGB)
            for box in results.xyxy[0]:
                if faceCount < faceLimit:
                    faceCount += 1 
                    if box[5] == 1:
                        className = "Male:"
                        bgr =(230, 216, 173)
                    elif box[5] == 0:
                        className = "Female:"
                        bgr =(203, 192, 255)
                    
                    conf = math.floor(box[4] * 100)
                    xB = int(box[2])
                    xA = int(box[0])
                    yB = int(box[3])
                    yA = int(box[1])
                        
                    cv2.rectangle(frame, (xA, yA), (xB, yB), (bgr), 4)
                    cv2.rectangle(frame, (xA, yA-50), (xA+180, yA), (bgr), -1)
                    cv2.putText(frame, str(conf), (xA + 130, yA-13), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=(255, 255, 255))
                    cv2.putText(frame, className, (xA, yA-15), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=(255, 255, 255))
                else:
                    break

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')


app.run(host='0.0.0.0', port=50000, debug=True)