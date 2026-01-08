# Indian Sign Language Translation using DETR and NLP

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-ee4c2c)
![DETR](https://img.shields.io/badge/Model-DETR-orange)
![NLP](https://img.shields.io/badge/Task-Translation-brightgreen)

This repository implements an end-to-end pipeline for translating **Indian Sign Language (ISL)** into natural English text. By combining **DETR (DEtection TRansformer)** for precise gesture detection and **Natural Language Processing (NLP)** for linguistic mapping, this project aims to facilitate seamless communication for the speech and hearing-impaired community.

---

## 📖 Table of Contents
* [Overview](#-overview)
* [System Architecture](#-system-architecture)
* [Features](#-features)
* [Installation](#-installation)
* [Usage](#-usage)
* [Dataset](#-dataset)
* [Results](#-results)
* [Contributing](#-contributing)

---

## 🚀 Overview
Indian Sign Language is visually rich and grammatically distinct from spoken English. This project addresses the translation challenge in two phases:
1. **Detection:** Using a Transformer-based object detection model (DETR) to identify signs (glosses) in video frames.
2. **Translation:** Using an NLP sequence-to-sequence approach to convert identified glosses into grammatically correct English sentences.

## 🏗 System Architecture
The pipeline follows a modular approach:



1. **Backbone (ResNet-50):** Extracts high-level features from input frames.
2. **Transformer Encoder-Decoder (DETR):** Predicts bounding boxes and class labels for specific ISL signs.
3. **Sequence Aggregator:** Tracks detected signs over time to form a "gloss sequence."
4. **Language Translation Model:** Maps the gloss sequence to a natural language output.

## ✨ Features
- **Transformer-based Detection:** Benefits from global context provided by DETR, reducing false positives in complex backgrounds.
- **End-to-End Workflow:** Supports everything from raw video input to final text output.
- **Customizable Backbone:** Easily switch between ResNet-50, ResNet-101, or other backbones.
- **Real-time Potential:** Optimized for high-throughput inference.

## 🛠 Installation

### 1. Clone the repo

git clone [https://github.com/CarolDsillva/Indian-Sign-Language-Translation-using-DETR-and-NLP.git](https://github.com/CarolDsillva/Indian-Sign-Language-Translation-using-DETR-and-NLP.git)
cd Indian-Sign-Language-Translation-using-DETR-and-NLP

### 2. Setup Environment
# Create a virtual environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

### 3. Download Weights
Place your pre-trained detr_model.pth and NLP model weights in the weights/ directory.
