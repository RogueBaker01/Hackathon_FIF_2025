import React, { useState, useEffect } from 'react';
import { Line, Bar } from 'react-chartjs-2';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faLightbulb, faPlug, faThermometerHalf, faUserShield } from '@fortawesome/free-solid-svg-icons';
import './Dashboard.css';

const Dashboard = () => {
  const [systemData, setSystemData] = useState(null);
  const [securityLog, setSecurityLog] = useState([]);
  const [darkMode, setDarkMode] = useState(false);

  // Configuración del tema
  const theme = {
    light: {
      bg: '#f8f9fa',
      cardBg: '#ffffff',
      text: '#2c3e50',
      primary: '#3498db',
      secondary: '#2ecc71'
    },
    dark: {
      bg: '#2c3e50',
      cardBg: '#34495e',
      text: '#ecf0f1',
      primary: '#2980b9',
      secondary: '#27ae60'
    }
  };

  useEffect(() => {
    const ws = new WebSocket(process.env.REACT_APP_WS_URL);
    
    ws.onmessage = (event) => {
      const newData = JSON.parse(event.data);
      setSystemData(newData);
    };

    fetch(`${process.env.REACT_APP_API_URL}/api/security-log`)
      .then(res => res.json())
      .then(data => setSecurityLog(data.log));

    return () => ws.close();
  }, []);

  // Configuración de gráficos
  const energyChartData = {
    labels: ['Iluminación', 'Computadoras', 'Climatización'],
    datasets: [{
      label: 'Consumo Energético (kWh)',
      data: systemData ? [
        systemData.energy_usage.lights.toFixed(2),
        systemData.energy_usage.computers.toFixed(2),
        systemData.energy_usage.hvac.toFixed(2)
      ] : [],
      backgroundColor: [
        theme[darkMode ? 'dark' : 'light'].primary,
        theme[darkMode ? 'dark' : 'light'].secondary,
        '#e74c3c'
      ]
    }]
  };

  return (
    <div className="dashboard" style={{ backgroundColor: theme[darkMode ? 'dark' : 'light'].bg }}>
      <header className="dashboard-header">
        <h1>Smart Office Dashboard</h1>
        <button 
          onClick={() => setDarkMode(!darkMode)}
          className="theme-toggle"
        >
          {darkMode ? '🌞 Light' : '🌙 Dark'}
        </button>
      </header>

      <div className="main-grid">
        {/* Tarjeta de Ocupación */}
        <div className="dashboard-card" style={{ backgroundColor: theme[darkMode ? 'dark' : 'light'].cardBg }}>
          <FontAwesomeIcon icon={faUserShield} size="2x" />
          <h3>Ocupación en Tiempo Real</h3>
          <div className="metric-display">
            <span className="metric-value">{systemData?.occupancy || 0}</span>
            <span className="metric-label">personas</span>
          </div>
        </div>

        {/* Gráfico de Energía */}
        <div className="dashboard-card chart-container" style={{ backgroundColor: theme[darkMode ? 'dark' : 'light'].cardBg }}>
          <h3>Distribución de Energía</h3>
          <Bar 
            data={energyChartData}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              plugins: {
                legend: { position: 'top' }
              }
            }}
          />
        </div>

        {/* Tarjetas de Estado */}
        <div className="status-cards">
          <div className="status-card" style={{ backgroundColor: theme[darkMode ? 'dark' : 'light'].cardBg }}>
            <FontAwesomeIcon icon={faLightbulb} />
            <h4>Iluminación</h4>
            <p>{systemData?.light_level || 0}%</p>
          </div>

          <div className="status-card" style={{ backgroundColor: theme[darkMode ? 'dark' : 'light'].cardBg }}>
            <FontAwesomeIcon icon={faPlug} />
            <h4>Computadoras</h4>
            <p>{systemData?.safety_status?.computers || 'off'}</p>
          </div>

          <div className="status-card" style={{ backgroundColor: theme[darkMode ? 'dark' : 'light'].cardBg }}>
            <FontAwesomeIcon icon={faThermometerHalf} />
            <h4>Climatización</h4>
            <p>{systemData?.safety_status?.gas || 'off'}</p>
          </div>
        </div>

        {/* Historial de Seguridad */}
        <div className="dashboard-card security-log" style={{ backgroundColor: theme[darkMode ? 'dark' : 'light'].cardBg }}>
          <h3>Historial de Accesos</h3>
          <div className="log-entries">
            {securityLog.map((entry, index) => (
              <div 
                key={index} 
                className={`log-entry ${entry.access_granted ? 'granted' : 'denied'}`}
              >
                <span>{new Date(entry.timestamp).toLocaleTimeString()}</span>
                <strong>{entry.rfid}</strong>
                <span>{entry.access_granted ? '✅' : '❌'}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;