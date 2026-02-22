# Ajustes Generales

La página General en Configuración de Máquina contiene información básica de la máquina y ajustes de velocidad.

![Ajustes Generales](/screenshots/machine-general.png)

## Nombre de Máquina

Dale a tu máquina un nombre descriptivo. Esto ayuda a identificar la máquina en el menú desplegable de selección de máquina cuando tienes múltiples máquinas configuradas.

Ejemplos:
- "K40 del Taller"
- "Láser de Diodo del Garaje"
- "Ortur LM2 Pro"

## Velocidades y Aceleración

Estos ajustes controlan las velocidades máximas y aceleración para la planificación de movimiento y estimación de tiempo.

### Velocidad Máxima de Desplazamiento

La velocidad máxima para movimientos rápidos (sin corte). Esto se usa cuando el láser está apagado y la cabeza se está moviendo a una nueva posición.

- **Rango típico**: 2000-5000 mm/min
- **Propósito**: Planificación de movimiento y estimación de tiempo
- **Nota**: La velocidad real también está limitada por los ajustes de tu firmware

### Velocidad Máxima de Corte

La velocidad máxima permitida durante operaciones de corte o grabado.

- **Rango típico**: 500-2000 mm/min
- **Propósito**: Limita las velocidades de operación por seguridad
- **Nota**: Las operaciones individuales pueden usar velocidades más bajas

### Aceleración

La tasa a la que la máquina acelera y desacelera.

- **Rango típico**: 500-2000 mm/s²
- **Propósito**: Estimación de tiempo y planificación de movimiento
- **Nota**: Debe coincidir o ser menor que los ajustes de aceleración del firmware

:::tip
Comienza con valores de velocidad conservadores y aumenta gradualmente. Observa tu máquina en busca de saltos de correa, bloqueo de motores o pérdida de precisión de posicionamiento.
:::

## Ver También

- [Ajustes de Hardware](hardware) - Dimensiones de máquina y configuración de ejes
- [Ajustes de Dispositivo](device) - Conexión y ajustes de GRBL
