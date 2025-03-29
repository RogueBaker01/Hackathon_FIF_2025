import cv2
import time
import random
import asyncio
import numpy as np
from datetime import datetime, time
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from faker import Faker
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import uvicorn

# 1. Configuración inicial
app = FastAPI(title="EcoGob Office Manager", version="2.0")
fake = Faker()
np.random.seed(42)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Clases de Configuración del Sistema
class OfficeConfig(BaseModel):
    work_start: time = time(8, 0) # Hora inicio 8:00 AM
    work_end: time = time(18, 0) # Hora fin 6:00 PM
    authorized_rfid: set = {"CLEAN123", "MAINT456"}

class InfraredCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2()
        
    def detect_occupancy(self):
        ret, frame = self.cap.read()
        if not ret:
            return random.randint(0, 5)
            
        fg_mask = self.background_subtractor.apply(frame)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return len([c for c in contours if cv2.contourArea(c) > 500])

# 3. Simulador de Sensores y Dispositivos
class OfficeDeviceSimulator:
    def __init__(self):
        self.camera = InfraredCamera()
        self.rfid_reader = lambda: random.choice([None, "CLEAN123", "MAINT456", "INVALID789"])
        
    def generate_energy_data(self):
        return {
            "lights": random.uniform(0.5, 5.0),
            "computers": random.uniform(10.0, 150.0),
            "hvac": random.uniform(2.0, 25.0)
        }
    
    def generate_security_data(self):
        return {
            "water_valves": random.choices([0, 1], weights=[0.9, 0.1])[0],
            "gas_valves": random.choices([0, 1], weights=[0.95, 0.05])[0]
        }

# 4. Agentes Inteligentes Mejorados
class EnhancedLightningAgent:
    def __init__(self, config: OfficeConfig):
        self.config = config
        self.model = pipeline("text-classification", 
                            model="hf-internal-testing/tiny-random-BertForSequenceClassification")
        
    def calculate_lighting(self, occupancy: int, current_time: datetime):
        if not self.is_work_time(current_time):
            return 0
            
        input_text = f"{occupancy} {current_time.hour}"
        prediction = self.model(input_text)
        return int(prediction[0]['label'].split('_')[-1])
    
    def is_work_time(self, dt: datetime):
        return self.config.work_start <= dt.time() < self.config.work_end

class SafetyControlAgent:
    def __init__(self, config: OfficeConfig):
        self.config = config
        self.last_authorized_access = None
        
    def check_systems(self, current_time: datetime, rfid_status: bool):
        base_state = {
            "lights": "off",
            "computers": "shutdown",
            "water": "locked",
            "gas": "valve_closed"
        }
        
        if self.is_authorized_period(current_time, rfid_status):
            return {
                "lights": "auto",
                "computers": "normal",
                "water": "unlocked",
                "gas": "valve_open"
            }
        return base_state
    
    def is_authorized_period(self, dt: datetime, rfid_status: bool):
        return (self.is_work_time(dt) or 
               (rfid_status and self.within_grace_period(dt)))
    
    def is_work_time(self, dt: datetime):
        return self.config.work_start <= dt.time() < self.config.work_end
    
    def within_grace_period(self, dt: datetime):
        if self.last_authorized_access:
            return (dt - self.last_authorized_access).total_seconds() < 3600
        return False

# 5. Sistema Central
class OfficeAutomationSystem:
    def __init__(self):
        self.config = OfficeConfig()
        self.devices = OfficeDeviceSimulator()
        self.light_agent = EnhancedLightningAgent(self.config)
        self.safety_agent = SafetyControlAgent(self.config)
        self.last_update = datetime.now()
        self.security_log = []
        
    def update_systems(self):
        current_time = datetime.now()
        rfid = self.devices.rfid_reader()
        rfid_status = rfid in self.config.authorized_rfid if rfid else False
        
        # Control de iluminación
        occupancy = self.devices.camera.detect_occupancy()
        light_level = self.light_agent.calculate_lighting(occupancy, current_time)
        
        # Control de seguridad
        safety_status = self.safety_agent.check_systems(current_time, rfid_status)
        
        # Registrar eventos
        if rfid:
            log_entry = {
                "timestamp": current_time,
                "rfid": rfid,
                "access_granted": rfid_status,
                "systems_activated": safety_status
            }
            self.security_log.append(log_entry)
        
        return {
            "timestamp": current_time.isoformat(),
            "occupancy": occupancy,
            "light_level": light_level,
            "safety_status": safety_status,
            "energy_usage": self.devices.generate_energy_data()
        }

# 6. Configuración de la API
app.state.system = OfficeAutomationSystem()

@app.websocket("/ws/real-time")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            system_data = app.state.system.update_systems()
            await websocket.send_json(system_data)
            await asyncio.sleep(5)
    except Exception as e:
        print(f"WebSocket Error: {e}")

@app.get("/api/system-status")
async def get_system_status():
    return app.state.system.update_systems()

@app.get("/api/security-log")
async def get_security_log(limit: int = 10):
    return {"log": app.state.system.security_log[-limit:]}

class RFIDRequest(BaseModel):
    rfid_code: str

@app.post("/api/validate-rfid")
async def validate_rfid(request: RFIDRequest):
    is_valid = request.rfid_code in app.state.system.config.authorized_rfid
    if is_valid:
        app.state.system.safety_agent.last_authorized_access = datetime.now()
    return {"valid": is_valid}

# 7. Simulación en tiempo real
async def continuous_simulation():
    while True:
        app.state.system.update_systems()
        await asyncio.sleep(5)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(continuous_simulation())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)