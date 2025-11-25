# 🎵 AI Vocal Separator

> **Term Project 01 - AI Model to User Application**  
> A CLI-based application that separates mixed music tracks into individual stems (vocals, drums, bass, etc.) using the Spleeter AI model.

![Python](https://img.shields.io/badge/Python-3.8%20|%203.9%20|%203.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12.0-orange)
![Spleeter](https://img.shields.io/badge/Spleeter-2.3.2-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📖 Introduction

This project aims to help musicians and creators by providing an easy-to-use tool for extracting **vocals** or **instrumental tracks (MR)** from MP3 files. It utilizes the **Spleeter** deep learning model (by Deezer) to perform high-quality source separation.

### **Key Features**
- **2 Stems Separation**: Vocals / Accompaniment  
- **4 Stems Separation**: Vocals / Drums / Bass / Other  
- **5 Stems Separation**: Vocals / Drums / Bass / Piano / Other  
- **User-Friendly CLI**

---

## ⚙️ Environment & Prerequisites

### **Python Version**
- **3.8 ~ 3.10**  
  *(Python 3.11+ is NOT supported due to library conflicts)*

### **Required Tools**
- **FFmpeg**

#### Install FFmpeg

**Ubuntu**
```
sudo apt-get install ffmpeg
```

**Windows**
```
choco install ffmpeg
```

---

## 🚀 Installation

### **1. Clone the repository**
```
git clone https://github.com/KimLuna/EF2039_AI_termproject.git
cd EF2039_AI_termproject
```

### **2. Install dependencies**
```
pip install -r requirements.txt
```

---

## 💻 Usage

### **1. Basic Usage (Default: 2stems)**
```
python main.py
```

### **2. Select Specific Model**
```
python main.py --model 4stems
```

### **3. Select Input File**
```
python main.py --input my_favorite_song.mp3
```

### **4. Combine Options**
```
python main.py -m 5stems -i pop_song.mp3
```

---

## 📂 Output Structure

```
output/
├── test_song_2stems/
│   ├── vocals.wav
│   └── accompaniment.wav
└── my_song_4stems/
    ├── vocals.wav
    ├── drums.wav
    ├── bass.wav
    └── other.wav
```

---

## ⚠️ Troubleshooting

```
if 'TF_CONFIG' in os.environ:
    del os.environ['TF_CONFIG']
```

---

## 📜 License

This project follows the **MIT License**.
