# Ajustes de Láser

La página de Láser en Configuración de Máquina configura tu(s) cabezal(es) láser y sus propiedades.

![Ajustes de Láser](/screenshots/machine-laser.png)

## Cabezales Láser

Rayforge soporta máquinas con múltiples cabezales láser. Cada cabezal láser tiene su propia configuración.

### Añadir un Cabezal Láser

Haz clic en el botón **Añadir Láser** para crear una nueva configuración de cabezal láser.

### Propiedades del Cabezal Láser

Cada cabezal láser tiene los siguientes ajustes:

#### Nombre

Un nombre descriptivo para este cabezal láser.

Ejemplos:
- "Diodo 10W"
- "Tubo CO2"
- "Láser Infrarrojo"

#### Número de Herramienta

El índice de herramienta para este cabezal láser. Usado en código G con el comando T.

- Máquinas de un solo cabezal: Usar 0
- Máquinas multi-cabezal: Asignar números únicos (0, 1, 2, etc.)

#### Potencia Máxima

El valor de potencia máxima para tu láser.

- **Típico GRBL**: 1000 (rango S0-S1000)
- **Algunos controladores**: 255 (rango S0-S255)
- **Modo porcentaje**: 100 (rango S0-S100)

Este valor debería coincidir con el ajuste $30 de tu firmware.

#### Potencia de Enmarcado

El nivel de potencia usado para operaciones de enmarcado (delimitando sin
cortar).

- Establecer en 0 para deshabilitar el enmarcado
- Ajusta según tu láser y material

#### Potencia de Enfoque

El nivel de potencia usado cuando el modo enfoque está activado. El modo enfoque
enciende el láser a baja potencia para actuar como "puntero láser" para
posicionamiento.

- Establecer en 0 para deshabilitar la función de modo enfoque
- Usar para alineación visual y posicionamiento

:::tip Usando el Modo Enfoque
Haz clic en el botón de enfoque (icono de láser) en la barra de herramientas
para alternar el modo enfoque. El láser se encenderá a este nivel de potencia,
ayudándote a ver exactamente dónde está posicionado el láser. Consulta
[Posicionamiento de Piezas de Trabajo](../features/workpiece-positioning)
para más información.
:::

#### Tamaño del Punto

El tamaño físico de tu haz láser enfocado en milímetros.

- Ingresa ambas dimensiones X e Y
- La mayoría de los láseres tienen un punto circular (ej., 0.1 x 0.1)
- Afecta los cálculos de calidad de grabado

:::tip Midiendo el Tamaño del Punto
Para medir el tamaño de tu punto:
1. Dispara un pulso corto a baja potencia en un material de prueba
2. Mide la marca resultante con calibradores
3. Usa el promedio de múltiples mediciones
:::

## Ver También

- [Ajustes de Dispositivo](device) - Ajustes de modo láser de GRBL
- [Posicionamiento de Piezas de Trabajo](../features/workpiece-positioning) -
  Uso del modo enfoque y otros métodos de posicionamiento
