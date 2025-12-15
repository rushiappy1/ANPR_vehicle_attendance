import os
import json
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# CONFIG
DATA_ROOT = "/home/roshan/ASUS/ANPR_PIP_LINE/CNN_TrainingData_bw"
LABEL_FILE = "/home/roshan/ASUS/ANPR_PIP_LINE/CNN_TrainingData_bw/labels.txt"
OUTPUT_JSON = "predictions_crnn_simple.json"

IMG_HEIGHT = 48
IMG_WIDTH = 192
BATCH_SIZE = 16
NUM_EPOCHS = 111
VAL_SPLIT = 0.15
LR = 3e-4
WEIGHT_DECAY = 1e-2
PATIENCE = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK_IDX = 0
char_to_idx = {c: i + 1 for i, c in enumerate(alphabet)}  # 0 = blank
idx_to_char = {i + 1: c for i, c in enumerate(alphabet)}

# DATASET
class PlateCRNNDataset(Dataset):
    def __init__(self, root_dir: str, label_file: str,
                 img_size: Tuple[int, int] = (IMG_HEIGHT, IMG_WIDTH),
                 train: bool = True):
        self.root = root_dir
        self.train = train
        self.img_h, self.img_w = img_size

        with open(label_file, "r") as f:
            lines = [l.strip().split(maxsplit=1) for l in f]
        self.samples = [(p, t) for p, t in lines]

        base = [
            transforms.Resize((self.img_h, self.img_w)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]

        if train:
            aug = [
                transforms.RandomRotation(3),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.08, 0.08),
                    shear=4
                ),
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4
                ),
            ]
            self.transform = transforms.Compose(aug + base)
        else:
            self.transform = transforms.Compose(base)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, text = self.samples[idx]
        path = os.path.join(self.root, rel_path)
        img = Image.open(path).convert("L")
        img = self.transform(img)
        return img, text
    
# ENCODING / DECODING
def encode_texts(texts: List[str]):
    targets = []
    lengths = []
    for t in texts:
        t = t.strip().upper()
        indices = [char_to_idx[c] for c in t if c in char_to_idx]
        lengths.append(len(indices))
        targets.extend(indices)
    targets = torch.tensor(targets, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return targets, lengths


def ctc_greedy_decode(logits: torch.Tensor):
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

# SIMPLE CRNN (CNN + BiLSTM)

class SimpleCRNN(nn.Module):
    def __init__(self, img_h: int, n_classes: int, hidden: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),           # H/2, W/2
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),           # H/4, W/4
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)), # H/8, W/4
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d((2, 1), (2, 1)), # H/16, W/4
        )
        # compute feature dim automatically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, img_h, IMG_WIDTH)
            feats = self.cnn(dummy)
            _, c, h, w = feats.size()
            self.feat_h = h
            self.feat_c = c
            self.feat_dim = c * h

        self.rnn = nn.LSTM(
            input_size=self.feat_dim,
            hidden_size=hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.5,
        )
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):
        # x: [B, 1, H, W]
        feats = self.cnn(x)                      # [B, C, H', W']
        b, c, h, w = feats.size()
        feats = feats.permute(0, 3, 1, 2)        # [B, W', C, H']
        feats = feats.contiguous().view(b, w, c * h)  # [B, T, F]

        seq, _ = self.rnn(feats)                 # [B, T, 2H]
        logits = self.fc(seq)                    # [B, T, C]
        logits = logits.permute(1, 0, 2)         # [T, B, C]
        return logits
    
# EARLY STOPPING
class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0

    def step(self, val_loss):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# MAIN TRAINING
def main():
    full_dataset = PlateCRNNDataset(DATA_ROOT, LABEL_FILE, train=True)
    print("Total samples:", len(full_dataset))
    n_total = len(full_dataset)
    n_val = int(n_total * VAL_SPLIT)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True)

    num_classes = len(alphabet) + 1
    model = SimpleCRNN(IMG_HEIGHT, num_classes).to(DEVICE)

    criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    early_stopper = EarlyStopper(patience=PATIENCE, min_delta=0.01)

    best_val = float("inf")
    best_path = "crnn_simple_best.pth"

    for epoch in range(NUM_EPOCHS):
        # train 
        model.train()
        train_loss = 0.0
        n_train_samples = 0

        for imgs, texts in train_loader:
            imgs = imgs.to(DEVICE)
            targets, target_lengths = encode_texts(texts)
            targets = targets.to(DEVICE)
            target_lengths = target_lengths.to(DEVICE)

            logits = model(imgs)  # [T, B, C]
            input_lengths = torch.full(
                size=(logits.size(1),),
                fill_value=logits.size(0),
                dtype=torch.long,
                device=DEVICE,
            )

            log_probs = logits.log_softmax(2)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            bs = imgs.size(0)
            train_loss += loss.item() * bs
            n_train_samples += bs

        train_loss /= n_train_samples

        # val 
        model.eval()
        val_loss = 0.0
        n_val_samples = 0
        with torch.no_grad():
            for imgs, texts in val_loader:
                imgs = imgs.to(DEVICE)
                targets, target_lengths = encode_texts(texts)
                targets = targets.to(DEVICE)
                target_lengths = target_lengths.to(DEVICE)

                logits = model(imgs)
                input_lengths = torch.full(
                    size=(logits.size(1),),
                    fill_value=logits.size(0),
                    dtype=torch.long,
                    device=DEVICE,
                )
                log_probs = logits.log_softmax(2)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)

                bs = imgs.size(0)
                val_loss += loss.item() * bs
                n_val_samples += bs

        val_loss /= n_val_samples
        scheduler.step()

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | train CTC {train_loss:.4f} | val CTC {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            print("  -> new best")

        if early_stopper.step(val_loss):
            print("Early stopping triggered")
            break

    # inference to JSON
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    model.eval()

    results = []
    tfm_infer = PlateCRNNDataset(DATA_ROOT, LABEL_FILE, train=False).transform

    with torch.no_grad():
        for rel_path, _ in full_dataset.samples:
            path = os.path.join(DATA_ROOT, rel_path)
            img = Image.open(path).convert("L")
            img_t = tfm_infer(img).unsqueeze(0).to(DEVICE)

            logits = model(img_t)
            preds = ctc_greedy_decode(logits)[0]
            results.append({"image": rel_path, "plate": preds})

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved predictions to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

