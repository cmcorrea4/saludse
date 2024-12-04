import streamlit as st
import paho.mqtt.client as mqtt
import json
from datetime import datetime
import time
from collections import deque

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Monitoreo con API",
    page_icon="🌡️",
    layout="wide"
)

# Configuración MQTT
MQTT_BROKER = "157.230.214.127"
MQTT_PORT = 1883
MQTT_TOPIC = "sensor_st"

# Inicialización de variables en session state
if 'sensor_data' not in st.session_state:
    st.session_state.sensor_data = {
        'temp_data': deque(maxlen=100),
        'hum_data': deque(maxlen=100),
        'timestamps': deque(maxlen=100),
        'last_temp': 0,
        'last_hum': 0,
        'connected': False
    }

def get_mqtt_message():
    """Función para obtener un mensaje MQTT"""
    message_received = {"received": False, "payload": None}
    
    def on_message(client, userdata, message):
        try:
            payload = json.loads(message.payload.decode())
            message_received["payload"] = payload
            message_received["received"] = True
            
            # Actualizar datos en session_state
            timestamp = datetime.now()
            st.session_state.sensor_data['temp_data'].append(payload.get('temperatura', 0))
            st.session_state.sensor_data['hum_data'].append(payload.get('humedad', 0))
            st.session_state.sensor_data['timestamps'].append(timestamp)
            st.session_state.sensor_data['last_temp'] = payload.get('temperatura', 0)
            st.session_state.sensor_data['last_hum'] = payload.get('humedad', 0)
            
        except Exception as e:
            st.error(f"Error al procesar mensaje: {e}")
    
    try:
        client = mqtt.Client()
        client.on_message = on_message
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.subscribe(MQTT_TOPIC)
        client.loop_start()
        
        timeout = time.time() + 5
        while not message_received["received"] and time.time() < timeout:
            time.sleep(0.1)
        
        client.loop_stop()
        client.disconnect()
        
        return message_received["payload"]
    
    except Exception as e:
        st.error(f"Error de conexión: {e}")
        return None

# Interfaz principal
st.title("Sistema de Monitoreo con API")

# Tabs para diferentes secciones
tab1, tab2 = st.tabs(["Dashboard", "API"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Control de Sensor")
        if st.button("Obtener Lectura"):
            with st.spinner('Obteniendo datos del sensor...'):
                sensor_data = get_mqtt_message()
                if sensor_data:
                    st.success("Datos recibidos")
                    st.metric("Temperatura", f"{sensor_data.get('temperatura', 'N/A')}°C")
                    st.metric("Humedad", f"{sensor_data.get('humedad', 'N/A')}%")
                else:
                    st.warning("No se recibieron datos del sensor")
    
    with col2:
        st.subheader("Últimas Lecturas")
        if len(st.session_state.sensor_data['timestamps']) > 0:
            st.metric("Última Temperatura", f"{st.session_state.sensor_data['last_temp']}°C")
            st.metric("Última Humedad", f"{st.session_state.sensor_data['last_hum']}%")
        else:
            st.info("No hay lecturas disponibles")

with tab2:
    st.subheader("API Endpoints")
    
    # Endpoint actual
    st.markdown("### GET /sensor/actual")
    current_data = {
        "timestamp": datetime.now().isoformat(),
        "temperatura": st.session_state.sensor_data['last_temp'],
        "humedad": st.session_state.sensor_data['last_hum']
    }
    st.json(current_data)
    
    # Endpoint historial
    st.markdown("### GET /sensor/historial")
    if len(st.session_state.sensor_data['timestamps']) > 0:
        history_data = [{
            "timestamp": ts.isoformat(),
            "temperatura": temp,
            "humedad": hum
        } for ts, temp, hum in zip(
            st.session_state.sensor_data['timestamps'],
            st.session_state.sensor_data['temp_data'],
            st.session_state.sensor_data['hum_data']
        )]
        st.json(history_data[-5:])  # Mostrar últimos 5 registros
    else:
        st.json([])

# Sidebar con información
with st.sidebar:
    st.subheader("Información de Conexión")
    st.code(f"""
    MQTT Broker: {MQTT_BROKER}
    Puerto: {MQTT_PORT}
    Tópico: {MQTT_TOPIC}
    """)
    
    st.subheader("Formato de Datos")
    st.code("""
    {
        "temperatura": 25.5,
        "humedad": 60
    }
    """)

# Auto-actualización
if st.session_state.sensor_data['last_temp'] > 0:
    time.sleep(2)
    st.rerun()

