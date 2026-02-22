# Configuración Inicial

Después de instalar Rayforge, necesitarás configurar tu cortador o grabador láser. Esta guía te guiará en la creación de tu primera máquina y el establecimiento de una conexión.

## Paso 1: Iniciar Rayforge

Inicia Rayforge desde el menú de aplicaciones o ejecutando `rayforge` en una terminal. Deberías ver la interfaz principal con un lienzo vacío.

## Paso 2: Crear una Máquina

Ve a **Configuración → Máquinas** o presiona <kbd>ctrl+coma</kbd> para abrir el diálogo de configuración, luego selecciona la página **Máquinas**.

Haz clic en **Añadir Máquina** para crear una nueva máquina. Puedes:

1. **Elegir un perfil integrado** - Selecciona entre plantillas de máquina predefinidas
2. **Seleccionar "Personalizada"** - Comienza con una configuración en blanco

Después de seleccionar, se abre el diálogo de Configuración de Máquina para tu nueva máquina.

![Configuración de Máquina](/screenshots/application-machines.png)

## Paso 3: Configurar Ajustes Generales

La página **General** contiene información básica de la máquina, selección de controlador y configuración de conexión.

![Ajustes Generales](/screenshots/machine-general.png)

### Información de la Máquina

1. **Nombre de Máquina**: Dale a tu máquina un nombre descriptivo (ej., "K40 Laser", "Ortur LM2")

### Selección de Controlador

Selecciona el controlador apropiado para tu dispositivo del menú desplegable:

- **GRBL Serial** - Para dispositivos GRBL conectados vía USB/puerto serie
- **GRBL Network** - Para dispositivos GRBL con conectividad WiFi/Ethernet
- **Smoothie** - Para dispositivos basados en Smoothieware

### Configuración del Controlador

Dependiendo del controlador seleccionado, configura los parámetros de conexión:

#### GRBL Serial (USB)

1. **Puerto**: Elige tu dispositivo del menú desplegable (ej., `/dev/ttyUSB0` en Linux, `COM3` en Windows)
2. **Velocidad de Transmisión**: Selecciona `115200` (estándar para la mayoría de dispositivos GRBL)

:::info
Si tu dispositivo no aparece en la lista, verifica que esté conectado y que tengas los permisos necesarios. En Linux, puede que necesites añadir tu usuario al grupo `dialout`.
:::

#### GRBL Network / Smoothie (WiFi/Ethernet)

1. **Host**: Ingresa la dirección IP de tu dispositivo (ej., `192.168.1.100`)
2. **Puerto**: Ingresa el número de puerto (típicamente `23` o `8080`)

### Velocidades y Aceleración

Estos ajustes se usan para la estimación de tiempo de trabajo y optimización de trayectorias:

- **Velocidad Máxima de Desplazamiento**: Velocidad máxima de movimiento rápido
- **Velocidad Máxima de Corte**: Velocidad máxima de corte
- **Aceleración**: Usada para estimaciones de tiempo y cálculos de overscan

## Paso 4: Configurar Ajustes de Hardware

Cambia a la pestaña **Hardware** para configurar las dimensiones físicas de tu máquina.

![Ajustes de Hardware](/screenshots/machine-hardware.png)

### Dimensiones

- **Ancho**: Ingresa el ancho máximo de tu área de trabajo en milímetros
- **Alto**: Ingresa el alto máximo de tu área de trabajo en milímetros

### Ejes

- **Origen de Coordenadas (0,0)**: Selecciona dónde está ubicado el origen de tu máquina:
  - Abajo Izquierda (más común para GRBL)
  - Arriba Izquierda
  - Arriba Derecha
  - Abajo Derecha

### Desplazamiento de Ejes (Opcional)

Configura los desplazamientos X e Y si tu máquina los requiere para un posicionamiento preciso.

## Paso 5: Conexión Automática

Rayforge se conecta automáticamente a tu máquina cuando se inicia la aplicación (si la máquina está encendida y conectada). No necesitas hacer clic manualmente en un botón de conexión.

El estado de conexión se muestra en la esquina inferior izquierda de la ventana principal con un ícono de estado y etiqueta mostrando el estado actual (Conectado, Conectando, Desconectado, Error, etc.).

:::success ¡Conectado!
Si tu máquina muestra el estado "Conectado", ¡estás listo para empezar a usar Rayforge!
:::

## Opcional: Configuración Avanzada

### Múltiples Láseres

Si tu máquina tiene múltiples módulos láser (ej., diodo y CO2), puedes configurarlos en la página **Láser**.

![Ajustes de Láser](/screenshots/machine-laser.png)

Ver [Configuración de Láser](../machine/laser) para más detalles.

### Configuración de Cámara

Si tienes una cámara USB para alineación y posicionamiento, configúrala en la página **Cámara**.

![Ajustes de Cámara](/screenshots/machine-camera.png)

Ver [Integración de Cámara](../machine/camera) para más detalles.

### Configuración de Dispositivo

La página **Dispositivo** te permite leer y modificar configuraciones de firmware directamente en tu dispositivo conectado (como parámetros GRBL). Esta es una función avanzada y debe usarse con precaución.

:::warning
¡Editar la configuración del dispositivo puede ser peligroso y puede dejar tu máquina inoperativa si se aplican valores incorrectos!
:::

---

## Solución de Problemas de Conexión

### Dispositivo No Encontrado

- **Linux (Serial)**: Añade tu usuario al grupo `dialout`:
  ```bash
  sudo usermod -a -G dialout $USER
  ```
  Cierra sesión y vuelve a entrar para que los cambios surtan efecto.

- **Paquete Snap**: Asegúrate de haber otorgado permisos de puerto serie:
  ```bash
  sudo snap connect rayforge:serial-port
  ```

- **Windows**: Revisa el Administrador de Dispositivos para confirmar que el dispositivo es reconocido y anota el número de puerto COM.

### Conexión Rechazada

- Verifica que la dirección IP y el número de puerto sean correctos
- Asegúrate de que tu máquina esté encendida y conectada a la red
- Revisa la configuración del firewall si usas conexión de red

### La Máquina No Responde

- Prueba con una velocidad de transmisión diferente (algunos dispositivos usan `9600` o `57600`)
- Revisa si hay cables sueltos o conexiones deficientes
- Apaga y enciende tu cortador láser y vuelve a intentarlo

Para más ayuda, ver [Problemas de Conexión](../troubleshooting/connection).

---

**Siguiente:** [Guía de Inicio Rápido →](quick-start)
