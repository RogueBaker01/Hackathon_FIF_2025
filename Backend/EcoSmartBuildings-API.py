import time
import random
import asyncio
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from faker import Faker
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import uvicorn

# 1. Configuración inicial
app = FastAPI(title="EcoGob-Qro API", version="1.0")
fake = Faker()
np.random.seed(42)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Simulador de Sensores
class SensorSimulator:
    def __init__(self, building_id):
        self.building_id = building_id

    def generate_light_data(self):
        return {
            "ocupacion": random.randint(0, 1),
            "lux": random.randint(0, 2000),
            "hora": fake.time_object().hour
        }
    
    def generate_climate_data(self):
        return {
            "temp_interna": round(random.uniform(15.0, 35.0), 1),
            "temp_externa": round(random.uniform(-5.0, 40.0), 1),
            "humedad": random.randint(20, 95)
        }
    
    def generate_gas_data(self):
        return {
            "presion": round(random.uniform(0.5, 2.5), 1),
            "flujo": round(random.uniform(0.0, 25.0), 1),
            "valvula_abierta": random.choice([0, 1])
        }
    
    def generate_electric_data(self):
        return {
            "consumo": round(random.uniform(5.0, 150.0), 2)
        }

# 3. Implementación de Agentes
class LightningAgent:
    def __init__(self):
        self.model = pipeline("text-classification", 
                            model="hf-internal-testing/tiny-random-BertForSequenceClassification")
        
    def analyze(self, data):
        input_text = f"{data['ocupacion']} {data['lux']} {data['hora']}"
        prediction = self.model(input_text)
        return {"intensidad": int(prediction[0]['label'].split('_')[-1])}

class ClimateAgent:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        X_train = np.random.rand(100, 3) * 30
        y_train = np.random.randint(0, 2, 100)
        self.model.fit(self.scaler.fit_transform(X_train), y_train)
        
    def analyze(self, data):
        X = self.scaler.transform([[data['temp_interna'], data['temp_externa'], data['humedad']]])
        return {"accion": "encender" if self.model.predict(X)[0] > 0.5 else "apagar"}

class ElectricAgent:
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(8, activation='relu', input_shape=(1,)),
            tf.keras.layers.Dense(4, activation='relu'),
            tf.keras.layers.Dense(8, activation='relu')
        ])
        self.model.compile(optimizer='adam', loss='mse')
        
    def analyze(self, data):
        historico = np.random.rand(100, 1) * 150
        self.model.fit(historico, historico, epochs=1, verbose=0)
        reconstruccion = self.model.predict([[data['consumo']]])
        return {"anomalia": abs(reconstruccion[0][0] - data['consumo']) > 15}

class GasAgent:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-DistilBertModel")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "hf-internal-testing/tiny-random-DistilBertForSequenceClassification")
        
    def analyze(self, data):
        inputs = self.tokenizer(
            f"{data['presion']} {data['flujo']} {data['valvula_abierta']}", 
            return_tensors="pt"
        )
        logits = self.model(**inputs).logits
        return {"alerta": "ALERTA" if logits.argmax().item() == 1 else "OK"}

# 4. Sistema Principal y API
class SmartBuildingSystem:
    def __init__(self):
        self.sensor = SensorSimulator("EDIF_001")
        self.agents = {
            "light": LightningAgent(),
            "climate": ClimateAgent(),
            "electric": ElectricAgent(),
            "gas": GasAgent()
        }
        self.current_state = {}
        self.alert_history = []

    def get_status(self):
        return {
            "timestamp": time.time(),
            "sensors": self.current_state,
            "alerts": self.alert_history[-5:]
        }

app.state.system = SmartBuildingSystem()

@app.get("/api/status")
async def get_status():
    return app.state.system.get_status()

@app.websocket("/ws/real-time")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = app.state.system.get_status()
            await websocket.send_json(data)
            await asyncio.sleep(5)
    except Exception as e:
        print(f"Error de WebSocket: {e}")

async def simulate_sensors():
    while True:
        for sensor_type in ["light", "climate", "electric", "gas"]:
            data = getattr(app.state.system.sensor, f"generate_{sensor_type}_data")()
            analysis = app.state.system.agents[sensor_type].analyze(data)
            
            app.state.system.current_state[sensor_type] = {
                "data": data,
                "analysis": analysis
            }

            if (sensor_type == "electric" and analysis["anomalia"]) or \
               (sensor_type == "gas" and analysis["alerta"] == "ALERTA"):
                alert = {
                    "type": sensor_type,
                    "message": f"Alerta en {sensor_type}: {analysis}",
                    "timestamp": time.time()
                }
                app.state.system.alert_history.append(alert)
        
        await asyncio.sleep(5)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(simulate_sensors())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)