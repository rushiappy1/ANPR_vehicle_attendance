import os
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms

#CONFIG 
MODEL_PATH = os.path.join(os.getcwd(),"Models_CRNN/34e_crnn_simple_best.pth")
IMG_FOLDER = os.path.join("Temp")
OUT_JSON = "crnn_results.json"

IMG_HEIGHT = 48
IMG_WIDTH = 192
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK_IDX = 0
idx_to_char = {i + 1: c for i, c in enumerate(alphabet)}

#  EXACT SAME MODEL AS TRAINING 
class SimpleCRNN(nn.Module):
    def __init__(self, img_h: int, n_classes: int, hidden: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d((2, 1), (2, 1)),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, img_h, IMG_WIDTH)
            feats = self.cnn(dummy)
            _, c, h, _ = feats.size()
            self.feat_dim = c * h  # MUST be 1536

        self.rnn = nn.LSTM(
            input_size=self.feat_dim,
            hidden_size=hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):
        feats = self.cnn(x)
        b, c, h, w = feats.size()
        feats = feats.permute(0, 3, 1, 2)
        feats = feats.contiguous().view(b, w, c * h)
        seq, _ = self.rnn(feats)
        logits = self.fc(seq)
        logits = logits.permute(1, 0, 2)  # [T, B, C]
        return logits
        
# CTC DECODE (SAME AS TRAINING) 
def ctc_greedy_decode(logits):
    log_probs = logits.log_softmax(2)
    _, preds = log_probs.max(2)
    preds = preds.transpose(0, 1)

    results = []
    for seq in preds:
        prev = BLANK_IDX
        chars = []
        for p in seq.tolist():
            if p != BLANK_IDX and p != prev:
                chars.append(idx_to_char.get(p, ""))
            prev = p
        results.append("".join(chars))
    return results

#  INFERENCE TRANSFORM (IDENTICAL) 
infer_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def run_inference():
    model = SimpleCRNN(IMG_HEIGHT, len(alphabet) + 1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    results = []

    with torch.no_grad():
        for name in sorted(os.listdir(IMG_FOLDER)):
            if not name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            path = os.path.join(IMG_FOLDER, name)
            img = Image.open(path).convert("L")
            img_t = infer_transform(img).unsqueeze(0).to(DEVICE)

            logits = model(img_t)
            plate = ctc_greedy_decode(logits)[0]

            print(f"{name} → {plate}")
            results.append({
                "image": name,
                "prediction": plate
            })
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {OUT_JSON}")

if __name__ == "__main__":
    run_inference()

