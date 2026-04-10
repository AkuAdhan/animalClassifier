from flask import Flask, request, jsonify, render_template
from torchvision import models
import torch
from torch import nn
from torchvision.transforms import transforms
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["cat", "dog", "wild"]

val_test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Arsitektur SAMA PERSIS dengan notebook
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.base_model = models.mobilenet_v2(weights=None)
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, len(CLASS_NAMES))
        )

    def forward(self, x):
        return self.base_model(x)

model = Net().to(device)

MODEL_PATH = "animal_classifier_optimized.pth"
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully!")
else:
    print(f"WARNING: Model file '{MODEL_PATH}' not found.")
    model.eval()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = val_test_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probs   = torch.softmax(outputs, dim=1)[0]

        results = [
            {"label": CLASS_NAMES[i], "confidence": round(probs[i].item() * 100, 2)}
            for i in range(len(CLASS_NAMES))
        ]
        results.sort(key=lambda x: x["confidence"], reverse=True)

        img_b64 = base64.b64encode(image_bytes).decode("utf-8")

        return jsonify({
            "prediction": results[0]["label"],
            "confidence": results[0]["confidence"],
            "all_scores": results,
            "image_b64": img_b64
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
