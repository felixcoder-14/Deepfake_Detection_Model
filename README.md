
# 🕵️‍♂️ Deepfake Detection Model

### Classify real vs. fake (AI-generated) faces using Deep Learning

This repository contains an end-to-end deepfake detection pipeline that performs:

✅ Face extraction
✅ Pre-processing
✅ Deep learning classification
✅ Interactive UI via **Gradio**

Built using **MTCNN** for face detection and **MobileNetV2 (Transfer Learning)** for classification.

---

## 🚀 Features

* Detects whether an image is **real or deepfake**
* Uses **MTCNN** to locate & crop facial regions
* Transfer Learning with **MobileNetV2**
* Data augmentation for improved training
* Lightweight & fast inference
* Easy-to-use **Gradio web app**
* Modular & extensible code

---

## 🧠 Tech Stack

| Component        | Library                          |
| ---------------- | -------------------------------- |
| Face Detection   | MTCNN                            |
| Frame Extraction | OpenCV                           |
| Model            | TensorFlow / Keras (MobileNetV2) |
| App UI           | Gradio                           |
| Data Handling    | NumPy / Pandas                   |
| Dataset          | FaceForensics / Kaggle           |

---

## 📁 Project Structure

```
.
├── src/
│   ├── extract_frames.py
│   ├── crop_faces.py
│   ├── train.py
│   ├── evaluate.py
│   ├── app.py
│
├── models/
│   ├── mobilenetv2_model.h5
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── README.md
├── requirements.txt
└── LICENSE
```

> Folder names can vary — adjust based on your actual structure.

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/felixcoder-14/Deepfake-Detection-Model-.git
cd Deepfake-Detection-Model-
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

We use the **FaceForensics / Deepfake Detection Challenge dataset (Kaggle)**.

➡️ Video frames are extracted using **OpenCV**
➡️ Faces are cropped using **MTCNN**
➡️ Images resized to **(224 × 224)**

---

## 🧩 Preprocessing Pipeline

1️⃣ **Video → Frames**
2️⃣ **MTCNN Face Detection & Cropping**
3️⃣ Normalize / Resize
4️⃣ Store into dataset folders (`real/`, `fake/`)

---

## 🏋️ Model Training

We use **MobileNetV2** (pretrained on ImageNet) and fine-tune for binary classification.

### Why MobileNetV2?

* Lightweight
* High accuracy
* Suitable for edge devices

Training script example:

```bash
python src/train.py
```

Outputs:

* `mobilenetv2_model.h5`
* Model logs

---

## ✅ Evaluation

Run evaluation:

```bash
python src/evaluate.py
```

Outputs:

* Accuracy / Loss
* Confusion matrix
* Predictions

---

## 🌐 Gradio App

Run:

```bash
python src/app.py
```

A web interface will open.
Upload an image → get **Real / Fake** probability.

Preview UI:

```
[ Upload IMG ] → Prediction %
```

---

## 🔍 Flow Summary

```
Video → Frames → Face Crop → Resize → Train CNN → Predict via Gradio
```

---

## 📈 Applications

* Media verification
* Cybersecurity
* Fraud prevention
* Journalism
* Law enforcement

---

## ⚠️ Limitations

* Occasionally misclassifies low-resolution faces
* Multi-face detection may require refinement
* Dataset diversity impacts performance

---

## 🔮 Future Improvements

* Real-time video deepfake detection
* Larger dataset support
* Mobile / On-device inference
* Ensemble deepfake detectors

---

## 📬 Results Summary

✅ Accurate deepfake detection via MobileNetV2
✅ Works in real-time for images
✅ Accessible UI with Gradio

---

## 🗂 Requirements

```
tensorflow
keras
opencv-python
mtcnn
numpy
pandas
gradio
```

Install:

```bash
pip install -r requirements.txt
```

---

## 👨‍💻 Author

**Muhammad Sameer**

---

## 📚 References

* FaceForensics++
* [https://www.tensorflow.org](https://www.tensorflow.org)
* [https://opencv.org](https://opencv.org)
* [https://github.com/ipazc/mtcnn](https://github.com/ipazc/mtcnn)
* [https://gradio.app](https://gradio.app)
* Kaggle Deepfake Dataset


