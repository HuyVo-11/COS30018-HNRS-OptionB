import os
import sys
import random
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
import torchvision.datasets as datasets
import torchvision.transforms as transforms

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from preprocessing.preprocessing import preprocess, preprocess_for_torch


def _resolve_runtime_path(name):
    cwd_path = os.path.abspath(name)
    if os.path.exists(cwd_path):
        return cwd_path
    return os.path.join(CURRENT_DIR, name)


MODEL_PATH = _resolve_runtime_path("model_combined.pth")
SYNTH_DATA_DIR = _resolve_runtime_path("synthetic_operators")
REAL_OPERATOR_DIR = _resolve_runtime_path(os.path.join("src", "model", "extracted_images"))
DATA_DIR = _resolve_runtime_path("data")

SAMPLES_PER_CLASS = 5000
NUM_CLASSES = 16
EPOCHS = 4
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
SAMPLES_PER_EPOCH = 64000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INDEX_TO_CHAR = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
    5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "+", 11: "-", 12: "*", 13: "/", 14: "(", 15: ")",
}
CHAR_TO_INDEX = {value: key for key, value in INDEX_TO_CHAR.items()}

REAL_OPERATOR_FOLDERS = {
    "+": 10,
    "-": 11,
    "times": 12,
    "forward_slash": 13,
    "(": 14,
    ")": 15,
}

SYNTH_OPERATOR_FOLDERS = {
    "plus": 10,
    "minus": 11,
    "mul": 12,
    "div": 13,
    "lparen": 14,
    "rparen": 15,
}


def _draw_plus(img, cx, cy, size, thickness):
    arm = size // 2
    cv2.line(img, (cx - arm, cy), (cx + arm, cy), 255, thickness)
    cv2.line(img, (cx, cy - arm), (cx, cy + arm), 255, thickness)


def _draw_minus(img, cx, cy, size, thickness):
    arm = size // 2
    cv2.line(img, (cx - arm, cy), (cx + arm, cy), 255, thickness)


def _draw_multiply(img, cx, cy, size, thickness):
    arm = size // 2
    cv2.line(img, (cx - arm, cy - arm), (cx + arm, cy + arm), 255, thickness)
    cv2.line(img, (cx - arm, cy + arm), (cx + arm, cy - arm), 255, thickness)


def _draw_divide(img, cx, cy, size, thickness):
    arm = size // 2
    cv2.line(img, (cx - arm, cy + arm), (cx + arm, cy - arm), 255, thickness)


def _draw_left_paren(img, cx, cy, size, thickness):
    cv2.ellipse(img, (cx, cy), (size // 4, size // 2), 0, 85, 275, 255, thickness)


def _draw_right_paren(img, cx, cy, size, thickness):
    cv2.ellipse(img, (cx, cy), (size // 4, size // 2), 0, -105, 105, 255, thickness)


DRAW_FN = {
    "+": _draw_plus,
    "-": _draw_minus,
    "*": _draw_multiply,
    "/": _draw_divide,
    "(": _draw_left_paren,
    ")": _draw_right_paren,
}


def generate_operator_dataset():
    os.makedirs(SYNTH_DATA_DIR, exist_ok=True)

    for symbol, draw_fn in DRAW_FN.items():
        safe_name = {
            "+": "plus",
            "-": "minus",
            "*": "mul",
            "/": "div",
            "(": "lparen",
            ")": "rparen",
        }[symbol]
        folder = os.path.join(SYNTH_DATA_DIR, safe_name)
        os.makedirs(folder, exist_ok=True)

        print(f"[GEN] Generating {SAMPLES_PER_CLASS} samples for '{symbol}'")
        for i in range(SAMPLES_PER_CLASS):
            img = np.zeros((28, 28), dtype=np.uint8)
            cx = 14 + np.random.randint(-3, 4)
            cy = 14 + np.random.randint(-3, 4)
            size = np.random.randint(10, 20)
            thickness = np.random.randint(1, 4)

            draw_fn(img, cx, cy, size, thickness)

            angle = np.random.uniform(-20, 20)
            matrix = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
            img = cv2.warpAffine(img, matrix, (28, 28))

            noise = np.random.normal(0, 10, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            shift = np.random.randint(-15, 16)
            img = np.clip(img.astype(np.int16) + shift, 0, 255).astype(np.uint8)

            cv2.imwrite(os.path.join(folder, f"{i:05d}.png"), img)

    print("[GEN] Synthetic operator dataset generation complete.")


class TorchVisionDigitDataset(Dataset):
    def __init__(self, dataset, transform=None, fix_orientation=False):
        self.dataset = dataset
        self.transform = transform
        self.fix_orientation = fix_orientation

        raw_targets = dataset.targets
        if hasattr(raw_targets, "tolist"):
            raw_targets = raw_targets.tolist()
        self.labels = [int(label) for label in raw_targets]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        array = np.array(image, dtype=np.uint8)

        if self.fix_orientation:
            # EMNIST digits are transposed relative to MNIST.
            array = np.ascontiguousarray(array.T)

        tensor = torch.tensor(array, dtype=torch.float32).unsqueeze(0) / 255.0
        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor, int(label)


class ImageFolderCharacterDataset(Dataset):
    def __init__(
        self,
        root_dir,
        folder_to_class,
        transform=None,
        split="train",
        train_ratio=0.9,
        seed=42,
    ):
        self.transform = transform
        self.samples = []
        self.labels = []

        splitter = random.Random(seed)
        for folder_name, class_idx in folder_to_class.items():
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            files = [
                os.path.join(folder_path, name)
                for name in os.listdir(folder_path)
                if name.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            files.sort()
            splitter.shuffle(files)

            if split in {"train", "val"}:
                cutoff = max(1, int(len(files) * train_ratio))
                selected = files[:cutoff] if split == "train" else files[cutoff:]
            else:
                selected = files

            for path in selected:
                self.samples.append((path, class_idx))
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        normalized = preprocess(image)
        tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0) / 255.0
        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor, int(label)


class OperatorCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def _build_train_transform():
    return transforms.Compose([
        transforms.RandomAffine(
            degrees=18,
            translate=(0.12, 0.12),
            scale=(0.9, 1.1),
            shear=8,
            fill=0,
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.25, fill=0),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))],
            p=0.25,
        ),
    ])


def _build_training_datasets():
    transform = _build_train_transform()

    mnist_train = TorchVisionDigitDataset(
        datasets.MNIST(root=DATA_DIR, train=True, download=True),
        transform=transform,
        fix_orientation=False,
    )
    emnist_train = TorchVisionDigitDataset(
        datasets.EMNIST(root=DATA_DIR, split="digits", train=True, download=True),
        transform=transform,
        fix_orientation=True,
    )
    real_operator_train = ImageFolderCharacterDataset(
        REAL_OPERATOR_DIR,
        REAL_OPERATOR_FOLDERS,
        transform=transform,
        split="train",
    )
    synthetic_operator_train = ImageFolderCharacterDataset(
        SYNTH_DATA_DIR,
        SYNTH_OPERATOR_FOLDERS,
        transform=transform,
        split="train",
    )

    return [
        mnist_train,
        emnist_train,
        real_operator_train,
        synthetic_operator_train,
    ]


def _build_balanced_loader(datasets_list):
    labels = []
    for dataset in datasets_list:
        labels.extend(dataset.labels)

    label_counts = Counter(labels)
    class_weights = torch.ones(NUM_CLASSES, dtype=torch.float32)
    for label, count in label_counts.items():
        class_weights[label] = len(labels) / float(NUM_CLASSES * count)

    sample_weights = [float(class_weights[label]) for label in labels]
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=SAMPLES_PER_EPOCH,
        replacement=True,
    )

    loader = DataLoader(
        ConcatDataset(datasets_list),
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
    )
    return loader, class_weights


def train_combined_model():
    if not os.path.isdir(SYNTH_DATA_DIR) or len(os.listdir(SYNTH_DATA_DIR)) < 6:
        generate_operator_dataset()
    else:
        print("[INFO] Synthetic operator dataset already exists.")

    train_datasets = _build_training_datasets()
    train_loader, class_weights = _build_balanced_loader(train_datasets)

    model = OperatorCNN(num_classes=NUM_CLASSES).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print("[INFO] Loaded existing weights for fine-tuning.")
        except RuntimeError:
            print("[WARN] Existing weights do not match current architecture. Training from scratch.")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"[TRAIN] Starting training on {DEVICE}...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = logits.argmax(1)
            total += labels.size(0)
            correct += predictions.eq(labels).sum().item()

        accuracy = 100.0 * correct / max(1, total)
        avg_loss = total_loss / max(1, len(train_loader))
        print(f"  Epoch [{epoch + 1}/{EPOCHS}] Loss: {avg_loss:.4f} Acc: {accuracy:.2f}%")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[TRAIN] Model saved to '{MODEL_PATH}'")

    global _cached_model
    _cached_model = model.eval()


_cached_model = None


def _load_model():
    global _cached_model
    if _cached_model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file '{MODEL_PATH}' not found. Run `train_combined_model()` first."
            )

        model = OperatorCNN(num_classes=NUM_CLASSES).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        _cached_model = model

    return _cached_model


def predict_character(image_28x28):
    model = _load_model()
    tensor = preprocess_for_torch(image_28x28).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted = probabilities.max(1)

    return INDEX_TO_CHAR[predicted.item()], confidence.item()


if __name__ == "__main__":
    train_combined_model()
