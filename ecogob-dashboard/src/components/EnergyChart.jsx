import { Line } from 'react-chartjs-2';
import { Chart as ChartJS } from 'chart.js/auto';

const EnergyChart = ({ data }) => {
  const chartData = {
    labels: ['Luces', 'Computadoras', 'HVAC'],
    datasets: [{
      label: 'Consumo (kW)',
      data: data ? [data.lights, data.computers, data.hvac] : [],
      borderColor: '#4CAF50',
      tension: 0.4
    }]
  };

  return <Line data={chartData} />;
};