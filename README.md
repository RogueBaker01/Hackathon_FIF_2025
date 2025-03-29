# 🏢 EcoGob Office Manager - Smart Office Automation System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68.0-green.svg)](https://fastapi.tiangolo.com/)

Advanced office automation system combining IoT sensors, AI models, and real-time control for energy efficiency and security management.

![System Architecture](assets/architecture-diagram.png) <!-- Add actual diagram if available -->

## 📖 Table of Contents
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Endpoints](#-api-endpoints)
- [System Components](#-system-components)
- [License](#-license)

## ✨ Key Features
- **Real-time Occupancy Detection** using infrared camera and computer vision
- **AI-Powered Lighting Control** with TinyML model integration
- **RFID-based Security System** with access control
- **Energy Usage Simulation** for office equipment monitoring
- **Safety Systems Automation** (water/gas valves control)
- **WebSocket Real-time Updates** for system status

## 🔧 Tech Stack
**Core Framework:**  
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![UVicorn](https://img.shields.io/badge/UVicorn-499848?style=flat&logo=uvicorn&logoColor=white)

**Computer Vision:**  
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)

**AI/ML:**  
![Transformers](https://img.shields.io/badge/🤗_Transformers-FFD21E?style=flat)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)

**Data Processing:**  
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit_learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)

## 🛠️ Installation
```bash
# Clone repository
git clone https://github.com/RogueBaker01/Hackathon_FIF_2025.git
cd Hackathon_FIF_2025

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn main:app --reload
