# eco_gob_qro.py
import time
import random
import numpy as np
from faker import Faker
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from paho.mqtt import client as mqtt_client

# 1. Configuración Inicial
fake = Faker('es_ES')
np.random.seed(42)

# 2. Simulador de Sensores IoT
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
            "presion": round(random.uniform(0.5, 2.5)),  # bar
            "flujo": round(random.uniform(0.0, 25.0), 1),  # L/min
            "valvula_abierta": random.choice([0, 1])
        }
    
    def generate_electric_data(self):
        return {
            "consumo": round(random.uniform(5.0, 150.0), 2)  # kWh
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
        # Entrenamiento inicial con datos falsos
        X_train = np.random.rand(100, 3) * 30  # [temp_interna, temp_externa, humedad]
        y_train = np.random.randint(0, 2, 100)  # 0 = apagar, 1 = encender
        self.model.fit(self.scaler.fit_transform(X_train), y_train)
        
    def analyze(self, data):
        X = self.scaler.transform([[data['temp_interna'], data['temp_externa'], data['humedad']]])
        return {"accion": "encender" if self.model.predict(X)[0] > 0.5 else "apagar"}

class ElectricAgent:
    def __init__(self):
        # Autoencoder simple para detección de anomalías
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(8, activation='relu', input_shape=(1,)),
            tf.keras.layers.Dense(4, activation='relu'),
            tf.keras.layers.Dense(8, activation='relu')
        ])
        self.model.compile(optimizer='adam', loss='mse')
        
    def analyze(self, data):
        # Entrenamiento en vuelto con datos simulados
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

# 4. Sistema Principal
class SmartBuildingSystem:
    def __init__(self):
        self.sensor = SensorSimulator("EDIF_001")
        self.agents = {
            "light": LightningAgent(),
            "climate": ClimateAgent(),
            "electric": ElectricAgent(),
            "gas": GasAgent()
        }
        self.dashboard = {
            "ultima_actualizacion": None,
            "estado": {},
            "alertas": []
        }
        
    def simular_mqtt(self, topic, payload):
        """Simula la comunicación MQTT para el hackathon"""
        print(f" [MQTT] Mensaje recibido en {topic}: {payload}")
        
    def actualizar_dashboard(self, sensor_type, data, analysis):
        self.dashboard["ultima_actualizacion"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.dashboard["estado"][sensor_type] = {
            "data": data,
            "analysis": analysis
        }
        if any([("anomalia" in a and a["anomalia"]) or ("alerta" in a and a["alerta"] == "ALERTA") 
               for a in self.dashboard["estado"].values()]):
            self.dashboard["alertas"].append(f"Alerta detectada: {time.strftime('%H:%M:%S')}")

    def run(self, interval=5):
        """Ejecuta el sistema principal"""
        try:
            while True:
                # Generar y procesar datos de todos los sensores
                for sensor_type in ["light", "climate", "electric", "gas"]:
                    data = getattr(self.sensor, f"generate_{sensor_type}_data")()
                    analysis = self.agents[sensor_type].analyze(data)
                    
                    # Publicar via MQTT (simulado)
                    self.simular_mqtt(f"edificio/{sensor_type}", data)
                    
                    # Actualizar dashboard
                    self.actualizar_dashboard(sensor_type, data, analysis)
                
                # Mostrar estado actual
                print("\n" + "="*40)
                print(f"🏢 Estado del Sistema - {self.dashboard['ultima_actualizacion']}")
                print(f"💡 Iluminación: {self.dashboard['estado']['light']['analysis']['intensidad']}%")
                print(f"❄️ Clima: {self.dashboard['estado']['climate']['analysis']['accion'].upper()}")
                print(f"⚡ Consumo Eléctrico: {self.dashboard['estado']['electric']['data']['consumo']} kWh - {'🚨 ANOMALÍA' if self.dashboard['estado']['electric']['analysis']['anomalia'] else '✅ Normal'}")
                print(f"🔵 Gas: {self.dashboard['estado']['gas']['analysis']['alerta']}")
                
                if self.dashboard["alertas"]:
                    print("\n🚨 ALERTAS ACTIVAS:")
                    for alerta in self.dashboard["alertas"][-3:]:
                        print(f" - {alerta}")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n Sistema detenido")

# 5. Ejecución
if __name__ == "__main__":
    system = SmartBuildingSystem()
    system.run()