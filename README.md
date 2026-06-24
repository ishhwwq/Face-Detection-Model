# Face Detection Model

A custom deep learning model for real-time face detection with simultaneous **classification** and **bounding box regression**, built with TensorFlow/Keras and trained using augmented image data.

---

## Overview

This model tackles face detection as a multi-task learning problem — it predicts whether a face is present in an image (classification) and where it is (regression), jointly optimized through a combined loss function.

Built from scratch using a custom data collection pipeline (via OpenCV webcam capture), annotated with LabelMe, and trained end-to-end with Albumentations-based augmentation.

---

## Tech Stack

| Component | Library/Tool |
|---|---|
| Model Framework | TensorFlow / Keras |
| Annotation | LabelMe |
| Augmentation | Albumentations |
| Image Processing | OpenCV |
| Visualization | Matplotlib |
| Language | Python 3 |

---

## Architecture

- **Backbone:** VGG16 (pretrained on ImageNet, top removed, weights frozen as feature extractor)
- **Heads:**
  - Classification head → sigmoid output (face / no face)
  - Regression head → 4-unit output (bounding box coordinates)
- **Loss:** Combined total loss = classification loss + regression loss

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG16

def build_model():
    input_layer = Input(shape=(120, 120, 3))

    vgg = VGG16(include_top=False)(input_layer)

    # Classification head
    f1 = GlobalMaxPooling2D()(vgg)
    class_out = Dense(1, activation='sigmoid')(f1)

    # Regression head
    f2 = GlobalMaxPooling2D()(vgg)
    regress_out = Dense(4, activation='sigmoid')(f2)

    return Model(inputs=input_layer, outputs=[class_out, regress_out])

facetracker = build_model()
```

---

## Training Metrics

Trained for **10 epochs** on a custom dataset with 473 steps/epoch.

| Epoch | Total Loss | Class Loss | Regress Loss | Val Total Loss | Val Class Loss | Val Regress Loss |
|-------|-----------|------------|--------------|----------------|----------------|-----------------|
| 1 | 1.4716 | 0.3397 | 1.3017 | 1.1927 | 0.7208 | 0.8322 |
| 2 | 0.8125 | 0.2097 | 0.7077 | 0.5901 | 0.1759 | 0.5022 |
| 3 | 0.6488 | 0.1753 | 0.5612 | 0.2189 | 0.0221 | 0.2079 |
| 4 | 0.5606 | 0.1528 | 0.4842 | 0.3309 | 0.1412 | 0.2603 |
| 5 | 0.4648 | 0.1300 | 0.3998 | 0.1291 | 0.1039 | 0.0772 |
| 6 | 0.4110 | 0.1124 | 0.3548 | 0.0198 | 0.0217 | 0.0089 |
| 7 | 0.3572 | 0.0974 | 0.3085 | 0.0324 | 0.0068 | 0.0290 |
| 8 | 0.2881 | 0.0804 | 0.2479 | 1.1296 | 1.0840 | 0.5876 |
| 9 | 0.2521 | 0.0693 | 0.2174 | 0.0399 | 0.0085 | 0.0357 |
| 10 | 0.1952 | 0.0550 | 0.1676 | 0.0571 | 0.0091 | 0.0525 |

**Best validation performance:** Epoch 6 — Val Total Loss: `0.0198`, Val Class Loss: `0.0217`, Val Regress Loss: `0.0089`

Training loss dropped consistently from `1.47 → 0.19` over 10 epochs. Epoch 8 shows a validation spike (likely a hard batch), but the model recovered well by epoch 9.

---

## Project Structure

```
face-detection/
├── data/
│   ├── images/           # Raw captured images
│   ├── labels/           # LabelMe JSON annotations
│   └── augmented/        # Albumentations-augmented dataset
├── models/
│   └── facedetector.h5   # Saved model weights
├── notebooks/
│   └── train.ipynb       # Training notebook
├── detect.py             # Run inference on webcam/image
├── train.py              # Training script
└── README.md
```

---

## Setup

```bash
git clone https://github.com/ishhwwq/face-detection.git
cd face-detection
pip install tensorflow opencv-python matplotlib albumentations labelme
```

---

## Usage

### Collect training images (webcam)
```python
import cv2, os, uuid

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    img_name = os.path.join('data/images', f'{uuid.uuid4()}.jpg')
    cv2.imwrite(img_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
```

### Run inference
```python
import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('models/facedetector.h5')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    img = tf.image.resize(frame, (120, 120))
    img = img / 255.0
    pred = model.predict(np.expand_dims(img, axis=0))
    # pred[0] = class score, pred[1] = [x1, y1, x2, y2]
    if pred[0] > 0.5:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (pred[1][0] * [w, h, w, h]).astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
```

---

## Notes

- Epoch 8's validation spike is worth investigating — could be a bad augmentation batch or label misalignment in that fold
- Model generalizes well by epoch 9/10 with very low val regression loss (`0.0357`)
- For production use, consider training for more epochs with a learning rate scheduler

---

## Author

**Kaori** — [@ishhwwq](https://github.com/ishhwwq)
