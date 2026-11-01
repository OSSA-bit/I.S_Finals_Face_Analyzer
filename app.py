from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
import json
import base64

app = Flask(__name__)

RECOGNITION_THRESHOLD = 0.5
KNOWN_FACES_FILE = "known_faces.json"

# Initialize AI Models
face_detector = cv2.FaceDetectorYN.create(
    model="face_detection_yunet_2023mar.onnx",
    config="",
    input_size=(320, 320),
    score_threshold=0.6,
    nms_threshold=0.3,
    top_k=5000
)

face_recognizer = cv2.FaceRecognizerSF.create(
    model="face_recognition_sface_2021dec.onnx",
    config=""
)

known_faces = {}

def load_known_faces():
    global known_faces
    if os.path.exists(KNOWN_FACES_FILE):
        try:
            with open(KNOWN_FACES_FILE, 'r') as f:
                data = json.load(f)
                for name in data:
                    encoding = np.array(data[name]["encoding"])
                    if encoding.ndim > 1:
                        encoding = encoding.flatten()
                    data[name]["encoding"] = encoding
                known_faces = data
        except Exception as e:
            print(f"[SYSTEM ERROR] Failed to load known faces database: {e}")

def save_known_faces():
    try:
        data = {}
        for name in known_faces:
            encoding = known_faces[name]["encoding"]
            if encoding.ndim > 1:
                encoding = encoding.flatten()
            data[name] = {
                "encoding": encoding.tolist(),
                "count": known_faces[name].get("count", 1)
            }
        with open(KNOWN_FACES_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[SYSTEM ERROR] Failed to save faces database: {e}")

load_known_faces()

def base64_to_image(base64_string):
    img_data = base64_string.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "No image payload received"}), 400

    frame = base64_to_image(data['image'])
    height, width = frame.shape[:2]
    face_detector.setInputSize((width, height))
    detections = face_detector.detect(frame)

    results = []
    
    if detections[1] is not None:
        for detection in detections[1]:
            bbox = detection[:4].astype(np.int32).tolist()
            score = float(detection[4])
            
            name = "Unknown"
            confidence = 0.0
            
            if len(known_faces) > 0:
                try:
                    face_aligned = face_recognizer.alignCrop(frame, detection[:15])
                    encoding = face_recognizer.feature(face_aligned)
                    if encoding.ndim > 1:
                        encoding = encoding.flatten()
                    
                    best_match = None
                    best_confidence = 0
                    
                    for k_name, k_data in known_faces.items():
                        known_encoding = k_data["encoding"]
                        dot_product = np.dot(encoding, known_encoding)
                        norm_encoding = np.linalg.norm(encoding)
                        norm_known = np.linalg.norm(known_encoding)
                        
                        if norm_encoding > 0 and norm_known > 0:
                            similarity = dot_product / (norm_encoding * norm_known)
                            conf = (similarity + 1) / 2
                            
                            if conf > best_confidence and conf > RECOGNITION_THRESHOLD:
                                best_confidence = conf
                                best_match = k_name
                                
                    if best_match:
                        name = best_match
                        confidence = float(best_confidence)
                except Exception as e:
                    print(f"[MODEL ERROR] Face alignment/recognition failure: {e}")

            results.append({
                "box": bbox,
                "score": score,
                "name": name,
                "confidence": confidence
            })
            
    return jsonify({"faces": results})

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    name = data.get('name', '').strip()
    image_data = data.get('image')
    
    if not name or not image_data:
        return jsonify({"error": "Identity name or frame data missing"}), 400

    frame = base64_to_image(image_data)
    height, width = frame.shape[:2]
    face_detector.setInputSize((width, height))
    detections = face_detector.detect(frame)

    if detections[1] is None or len(detections[1]) == 0:
        return jsonify({"error": "No subject face detected in target frame"}), 400

    largest_face = max(detections[1], key=lambda d: d[2] * d[3])
    
    try:
        face_aligned = face_recognizer.alignCrop(frame, largest_face[:15])
        encoding = face_recognizer.feature(face_aligned)
        if encoding.ndim > 1:
            encoding = encoding.flatten()
        
        known_faces[name] = {
            "encoding": encoding,
            "count": 1
        }
        save_known_faces()
        return jsonify({"success": True, "message": f"Successfully enrolled profile for {name}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
