# Ajustes de Láser

La página de Láser en Configuración de Máquina configura tu(s) cabezal(es)
láser y sus propiedades.

![Ajustes de Láser](/screenshots/machine-laser.png)

## Cabezales Láser

Rayforge soporta máquinas con múltiples cabezales láser. Cada cabezal láser
tiene su propia configuración.

### Añadir un Cabezal Láser

Haz clic en el botón **Añadir Láser** para crear una nueva configuración de
cabezal láser.

### Propiedades del Cabezal Láser

Cada cabezal láser tiene los siguientes ajustes:

#### Nombre

Un nombre descriptivo para este cabezal láser.

Ejemplos:
- "Diodo 10W"
- "Tubo CO2"
- "Láser Infrarrojo"

#### Número de Herramienta

El índice de herramienta para este cabezal láser. Usado en código G con el
comando T.

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

#### Velocidad de Enmarcado

La velocidad a la que se mueve el cabezal láser durante el enmarcado. Se
establece por cabezal láser, por lo que si tu máquina tiene varios láseres con
diferentes características, puedes elegir una velocidad adecuada para cada uno.
Velocidades más lentas hacen que la ruta de enmarcado sea más fácil de seguir
visualmente.

#### Potencia de Enfoque

El nivel de potencia usado cuando el modo enfoque está activado. El modo
enfoque enciende el láser a baja potencia para actuar como "puntero láser" para
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

#### Color de Corte

El color usado para mostrar operaciones de corte para este láser en el lienzo y
la vista previa 3D. Esto te ayuda a distinguir visualmente qué láser realizará
cada operación de corte cuando trabajas con múltiples cabezales láser.

- Haz clic en la muestra de color para abrir un selector de color
- Elige un color que contraste bien con la vista previa de tu material
- Los colores predeterminados se asignan automáticamente

#### Color de Raster

El color usado para mostrar operaciones de raster/grabado para este láser en
el lienzo y la vista previa 3D.

- Haz clic en la muestra de color para abrir un selector de color
- Útil para diferenciar operaciones de raster de cortes
- Cada láser puede tener su propio color de raster distintivo

:::tip Flujos de Trabajo Multi-Láser
Al usar múltiples cabezales láser, asignar diferentes colores a cada láser
facilita ver qué operaciones serán realizadas por qué láser. Por ejemplo, usa
rojo para tu láser de corte principal y azul para un láser de grabado
secundario.
:::

#### Tipo de Láser

Elige el tipo de cabezal láser en el menú desplegable:

- **Diodo**: Láseres de diodo estándar (los más comunes en máquinas de afición)
- **CO2**: Láseres de tubo CO2
- **Fibra**: Láseres de fibra

Al seleccionar CO2 o Fibra, aparecen **ajustes PWM** adicionales (ver más
abajo). Para láseres de diodo, la sección PWM se oculta ya que no aplica.

#### Ajustes PWM

Cuando se selecciona un tipo de láser CO2 o Fibra, aparecen los siguientes
controles PWM:

- **Frecuencia PWM**: La frecuencia PWM predeterminada en Hz para este cabezal
  láser. Los valores típicos van de 500 Hz a varios kHz dependiendo de tu
  controlador y fuente de alimentación.
- **Frecuencia PWM máxima**: El límite superior para el ajuste de frecuencia.
  Esto evita introducir valores que tu hardware no puede manejar.
- **Ancho de pulso**: El ancho de pulso predeterminado en microsegundos. Controla
  cuánto tiempo permanece encendido cada pulso durante un ciclo.
- **Ancho de pulso mín/máx**: Límites para el ajuste del ancho de pulso.

Estos valores predeterminados se transfieren a tus pasos de operación, donde
pueden sobreescribirse por paso si es necesario.

#### Modelo 3D

Cada cabezal láser puede tener un modelo 3D asignado. Este modelo se renderiza
en la [vista 3D](../ui/3d-preview) y sigue la trayectoria durante la simulación.

Haz clic en la fila de selección de modelo para explorar los modelos disponibles.
Una vez seleccionado un modelo, puedes ajustar su escala, rotación (X/Y/Z) y
distancia focal para coincidir con tu cabezal láser físico.

## Ver También

- [Ajustes de Dispositivo](device) - Ajustes de modo láser de GRBL
- [Posicionamiento de Piezas de Trabajo](../features/workpiece-positioning) -
  Uso del modo enfoque y otros métodos de posicionamiento
