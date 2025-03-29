import { useState, useEffect } from 'react';
import { Box, Grid, Typography } from '@mui/material';
import { Line } from 'react-chartjs-2';
import SystemStatus from './components/SystemStatus';
import SecurityLog from './components/SecurityLog';

const App = () => {
  const [systemData, setSystemData] = useState(null);
  const [securityLog, setSecurityLog] = useState([]);

  // Configuración WebSocket
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/real-time');

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setSystemData(data);
    };

    return () => ws.close();
  }, []);

  // Cargar histórico inicial
  useEffect(() => {
    fetch('http://localhost:8000/api/security-log?limit=20')
      .then(res => res.json())
      .then(data => setSecurityLog(data.log));
  }, []);

  return (
    <Box sx={{ padding: 3 }}>
      <Typography variant="h3" gutterBottom>
        EcoGob Dashboard
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <EnergyChart data={systemData?.energy_usage} />
        </Grid>
        
        <Grid item xs={12} md={4}>
          <SystemStatus data={systemData} />
        </Grid>
        
        <Grid item xs={12} md={6}>
          <OccupancyGrid count={systemData?.occupancy} />
        </Grid>
        
        <Grid item xs={12} md={6}>
          <SecurityLog logs={securityLog} />
        </Grid>
      </Grid>
    </Box>
  );
};

export default App;