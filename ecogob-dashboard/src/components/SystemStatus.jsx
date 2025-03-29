import { Chip, Stack, Typography } from '@mui/material';
import { CheckCircle, Warning, Error } from '@mui/icons-material';

const SystemStatus = ({ data }) => {
  const status = data?.safety_status;
  
  return (
    <Stack spacing={2}>
      <Typography variant="h6">Estado del Sistema</Typography>
      
      <Chip
        icon={status?.lights === 'auto' ? <CheckCircle /> : <Error />}
        label={`Luces: ${status?.lights}`}
      />
      
      <Chip
        icon={status?.gas?.includes('open') ? <Warning /> : <CheckCircle />}
        label={`Gas: ${status?.gas}`}
      />
      
      <Chip
        label={`Ocupación: ${data?.occupancy || 0} personas`}
      />
    </Stack>
  );
};