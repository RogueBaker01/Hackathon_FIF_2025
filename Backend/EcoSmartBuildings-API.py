import cv2
import time
import random
import asyncio
import numpy as np
from datetime import datetime, time as dt_time
import logging
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from faker import Faker
from transformers import pipeline
import uvicorn

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EcoGobOfficeManager")

# 1. Configuración inicial
app = FastAPI(title="EcoGob Office Manager", version="2.1")
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
    work_start: dt_time = dt_time(8, 0)  # Hora inicio 8:00 AM
    work_end: dt_time = dt_time(18, 0)    # Hora fin 6:00 PM
    authorized_rfid: set = {"CLEAN123", "MAINT456"}

    def is_work_time(self, dt: datetime) -> bool:
        return self.work_start <= dt.time() < self.work_end

# 3. Cámara Infrarroja con manejo de recursos
class InfraredCamera:
    def __init__(self) -> None:
        self.cap = cv2.VideoCapture(0)
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2()
        if not self.cap.isOpened():
            logger.error("Error al abrir la cámara. Verifique la conexión.")
    
    def detect_occupancy(self) -> int:
        ret, frame = self.cap.read()
        if not ret:
            # Si la lectura falla, se retorna un valor simulado
            return random.randint(0, 5)
            
        fg_mask = self.background_subtractor.apply(frame)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return len([c for c in contours if cv2.contourArea(c) > 500])
    
    def release(self) -> None:
        if self.cap.isOpened():
            self.cap.release()

# 4. Simulador de Sensores y Dispositivos
class OfficeDeviceSimulator:
    def __init__(self, config: OfficeConfig) -> None:
        self.camera = InfraredCamera()
        self.config = config
        self.rfid_reader = lambda: random.choice([None, "CLEAN123", "MAINT456", "INVALID789"])
        
    def generate_energy_data(self, current_time: datetime, safety_status: dict) -> dict:
        """Genera datos de consumo simulados vinculados al estado del sistema"""
        is_work_time = self.config.is_work_time(current_time)
        
        base_consumption = {
            "security_systems": 0.8,  # Consumo de sistemas de seguridad
            "emergency_lights": 0.3    # Luces de emergencia
        }
        
        main_consumption = {
            "lights": random.uniform(0.5, 5.0) if safety_status["lights"] == "auto" else 0.1,
            "computers": random.uniform(10.0, 150.0) if safety_status["computers"] == "normal" else 2.5,
            "hvac": random.uniform(2.0, 25.0) if is_work_time else random.uniform(0.5, 5.0)
        }
        
        return {**main_consumption, **base_consumption}

    def generate_security_data(self) -> dict:
        return {
            "water_valves": random.choices([0, 1], weights=[0.9, 0.1])[0],
            "gas_valves": random.choices([0, 1], weights=[0.95, 0.05])[0]
        }

# 5. Agentes Inteligentes
class EnhancedLightningAgent:
    def __init__(self, config: OfficeConfig) -> None:
        self.config = config
        self.model = pipeline("text-classification", 
                              model="hf-internal-testing/tiny-random-BertForSequenceClassification")
        
    def calculate_lighting(self, occupancy: int, current_time: datetime) -> int:
        if not self.config.is_work_time(current_time):
            return 0  # Fuera de horario, luces apagadas
            
        input_text = f"{occupancy} {current_time.hour}"
        prediction = self.model(input_text)
        return int(prediction[0]['label'].split('_')[-1])

class SafetyControlAgent:
    def __init__(self, config: OfficeConfig) -> None:
        self.config = config
        self.last_authorized_access: datetime | None = None
        
    def check_systems(self, current_time: datetime, rfid_status: bool) -> dict:
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
    
    def is_authorized_period(self, dt: datetime, rfid_status: bool) -> bool:
        return self.config.is_work_time(dt) or (rfid_status and self.within_grace_period(dt))
    
    def within_grace_period(self, dt: datetime) -> bool:
        if self.last_authorized_access:
            return (dt - self.last_authorized_access).total_seconds() < 3600
        return False

# 6. Sistema Central de Automatización
class OfficeAutomationSystem:
    def __init__(self) -> None:
        self.config = OfficeConfig()
        self.devices = OfficeDeviceSimulator(self.config)
        self.light_agent = EnhancedLightningAgent(self.config)
        self.safety_agent = SafetyControlAgent(self.config)
        self.last_update = datetime.now()
        self.security_log: list[dict] = []
        
    def update_systems(self) -> dict:
        current_time = datetime.now()
        rfid = self.devices.rfid_reader()
        rfid_status = rfid in self.config.authorized_rfid if rfid else False
        
        occupancy = self.devices.camera.detect_occupancy()
        light_level = self.light_agent.calculate_lighting(occupancy, current_time)
        safety_status = self.safety_agent.check_systems(current_time, rfid_status)
        energy_usage = self.devices.generate_energy_data(current_time, safety_status)
        
        if rfid:
            log_entry = {
                "timestamp": current_time,
                "rfid": rfid,
                "access_granted": rfid_status,
                "systems_activated": safety_status
            }
            self.security_log.append(log_entry)
            logger.info(f"Registro RFID: {log_entry}")
        
        return {
            "timestamp": current_time.isoformat(),
            "occupancy": occupancy,
            "light_level": light_level,
            "safety_status": safety_status,
            "energy_usage": energy_usage
        }

# Inicializar el sistema central y almacenarlo en el estado de la app
app.state.system = OfficeAutomationSystem()

# 7. Endpoints de la API y WebSocket

@app.websocket("/ws/real-time")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            system_data = app.state.system.update_systems()
            await websocket.send_json(system_data)
            await asyncio.sleep(5)
    except Exception as e:
        logger.error(f"Error en WebSocket: {e}")
        await websocket.close()

@app.get("/api/system-status")
async def get_system_status():
    return app.state.system.update_systems()

@app.get("/api/security-log")
async def get_security_log(limit: int = 10):
    # Retornar los últimos registros de seguridad
    return {"log": app.state.system.security_log[-limit:]}

class RFIDRequest(BaseModel):
    rfid_code: str

@app.post("/api/validate-rfid")
async def validate_rfid(request: RFIDRequest):
    system = app.state.system
    is_valid = request.rfid_code in system.config.authorized_rfid
    current_time = datetime.now()
    
    # Crear entrada de log
    log_entry = {
        "timestamp": current_time,
        "rfid": request.rfid_code,
        "access_granted": is_valid,
        "systems_activated": system.safety_agent.check_systems(
            current_time, is_valid
        )
    }
    system.security_log.append(log_entry)
    logger.info(f"Validación manual RFID: {log_entry}")

    if is_valid:
        system.safety_agent.last_authorized_access = current_time
    return {"valid": is_valid}

# Simulación continua en segundo plano
async def continuous_simulation():
    while True:
        app.state.system.update_systems()
        await asyncio.sleep(5)

@app.on_event("startup")
async def startup_event():
    logger.info("Iniciando simulación continua...")
    asyncio.create_task(continuous_simulation())

@app.on_event("shutdown")
async def shutdown_event():
    # Liberar recursos de la cámara
    app.state.system.devices.camera.release()
    logger.info("Recursos liberados. Apagando la aplicación.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
