import io
import joblib
import cv2
import numpy as np
import pandas as pd
import torch
from flask import Flask, request, jsonify
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

app = Flask(__name__)

def preprocess(image):
    """Preprocess image by resizing while maintaining aspect ratio."""
    if image is None:
        raise ValueError("Invalid image provided.")
    
    k = min(1.0, 1024 / max(image.shape[0], image.shape[1]))
    img = cv2.resize(image, None, fx=k, fy=k, interpolation=cv2.INTER_LANCZOS4)
    
    return img


def create_buff(mask):
    buf = io.BytesIO()
    plt.imsave(buf, mask, cmap="gray", format="png")
    buf.seek(0)
    return buf

def access_stored_image(buffer):
    return Image.open(buffer)

def extract_silhouette_features_for_inference(image):
    image_array = np.array(image)
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    features = {'area': 0, 'perimeter': 0, 'aspect_ratio': 0, 'solidity': 0}
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h != 0 else 0.0
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area != 0 else 0.0
        
        features['area'] = area
        features['perimeter'] = perimeter
        features['aspect_ratio'] = aspect_ratio
        features['solidity'] = solidity
    
    return features

def make_deeplab(device):
    model = deeplabv3_resnet101(pretrained=True).to(device)
    model.eval()
    return model

def apply_deeplab(deeplab, img, device):
    preprocess_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess_transform(img).unsqueeze(0)
    with torch.no_grad():
        output = deeplab(input_tensor.to(device))['out'][0]
    return (output.argmax(0).cpu().numpy() == 15)

# Load models
model = joblib.load("multi_measurement_model.pkl")
scaler = joblib.load("scaler.pkl")
mean, scale = scaler.mean_, scaler.scale_
device = torch.device("cpu")
deeplab = make_deeplab(device)

@app.route('/predict', methods=['POST'])
def predict():
    try:

        front_file = request.files['front']
        left_file = request.files['left']
        height = float(request.form.get('height', 178))
        weight = float(request.form.get('weight', 87))
        gender = int(request.form.get('gender', 1))
        front_img = np.frombuffer(front_file.read(), np.uint8)
        front_img = cv2.imdecode(front_img, cv2.IMREAD_COLOR)

        left_img = np.frombuffer(left_file.read(), np.uint8)
        left_img = cv2.imdecode(left_img, cv2.IMREAD_COLOR)

        if front_img is None or left_img is None:
            return jsonify({"success": False, "error": "Invalid image file."})

        

        front_img = preprocess(front_img)
        mask_front = apply_deeplab(deeplab, front_img, device)
        stored_image_front = access_stored_image(create_buff(mask_front))
        features_front = extract_silhouette_features_for_inference(stored_image_front)

        left_img = preprocess(left_img)
        mask_left = apply_deeplab(deeplab, left_img, device)
        stored_image_left = access_stored_image(create_buff(mask_left))
        features_left = extract_silhouette_features_for_inference(stored_image_left)

        combined_features = {
            'area': features_front.get('area', 0),
            'perimeter': features_front.get('perimeter', 0),
            'aspect_ratio': features_front.get('aspect_ratio', 0),
            'solidity': features_front.get('solidity', 0),
            'area_left': features_left.get('area', 0),
            'perimeter_left': features_left.get('perimeter', 0),
            'aspect_ratio_left': features_left.get('aspect_ratio', 0),
            'solidity_left': features_left.get('solidity', 0)
        }
        
        df_inference = pd.DataFrame([combined_features])
        df_inference['weight_kg'] = weight
        df_inference['height'] = height
        df_inference['gender'] = gender
        
        norm_cols = ['area', 'height', 'weight_kg', 'perimeter', 'aspect_ratio', 'solidity',
                     'area_left', 'perimeter_left', 'aspect_ratio_left', 'solidity_left']
        for i, col in enumerate(norm_cols):
            df_inference[col] = (df_inference[col] - mean[i]) / scale[i]
        
        predictions = model.predict(df_inference)
        measurements = {
            "Neck": 2.64 + 1.94 * predictions[0, 0],
            "Waist": predictions[0, 1],
            "Hip": predictions[0, 2],
            "Chest": predictions[0, 3],
            "Ankle": predictions[0, 4],
            "Arm Length": predictions[0, 5],
            "Bicep": predictions[0, 6],
            "Calf": predictions[0, 7],
            "Forearm": predictions[0, 8],
            "Leg Length": predictions[0, 9],
            "Shoulder Breadth": predictions[0, 10],
            "Shoulder-to-Crotch": predictions[0, 11],
            "Thigh": predictions[0, 12]
        }
        return jsonify({"success": True, "measurements": measurements})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
