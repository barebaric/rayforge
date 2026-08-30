---
description:
  "Configura tu cortadora o grabadora láser por primera vez. Usa el asistente de configuración para
  crear tu máquina, luego conéctate y prepárate para cortar con Rayforge."
---

# Configuración Inicial

Después de instalar Rayforge, necesitarás configurar tu cortador o grabador láser. Esta guía te
guiará en la creación de tu primera máquina con el asistente de configuración y el establecimiento
de una conexión.

## Paso 1: Iniciar Rayforge

Inicia Rayforge desde el menú de aplicaciones o ejecutando `rayforge` en una terminal. En el primer
inicio — cuando aún no se ha configurado ninguna máquina real — el asistente de configuración se
abre automáticamente para que puedas configurar tu máquina sin buscar en los menús. (Siempre puedes
abrirlo después desde **Configuración → Máquinas → Add Machine**.)

## Paso 2: Crear una Máquina con el Asistente

Ve a **Configuración → Máquinas** o presiona <kbd>ctrl+coma</kbd> para abrir el diálogo de
configuración, luego selecciona la página **Máquinas**.

![Configuración de Máquina](/screenshots/app-settings-machines.webp)

Haz clic en **Add Machine** para abrir el selector de máquinas.

![Diálogo de Add Machine](/screenshots/app-settings-machines-add.webp)

### Comprobación de Permisos

Antes de que comience el descubrimiento, el asistente comprueba que Rayforge puede realmente abrir
tus puertos serie y cámaras. Si un dispositivo está presente pero falta el acceso, aparece primero
una **página de permisos** que explica cómo solucionarlo en tu plataforma:

- **Instalaciones Snap**: concede la interfaz `serial-port` (y la interfaz de cámara si es
  necesario) — los comandos exactos se muestran con un botón de copia con un solo clic.
- **Linux sin Snap**: añade tu usuario al grupo `dialout` para que el nodo del dispositivo serie sea
  accesible.

Una vez que el acceso está disponible, el asistente continúa automáticamente.

![Asistente — Comprobación de Permisos](/screenshots/config-wizard-permissions.webp)

### Descubrir Dispositivos Automáticamente

El asistente puede descubrir dispositivos por ti en lugar de requerir que elijas un punto de partida
y completes todo manualmente:

- **Dispositivos serie USB** se listan a medida que aparecen.
- **Dispositivos de red** se descubren vía mDNS: servidores OctoPrint y placas ESP3D aparecen junto
  a dispositivos serie USB.
- Los dispositivos descubridos se **emparejan con perfiles integrados** cuando se encuentra una
  coincidencia segura, por lo que a menudo solo puedes confirmar los ajustes detectados en lugar de
  introducirlos.
- GRBL selecciona automáticamente el dialecto de código G correcto a partir de las flags de
  compilación del firmware, y OctoPrint/Smoothieware se sondean a través de la red.
- Los dispositivos que ya has configurado se muestran como **solo lectura** para que no crees
  duplicados accidentalmente.

Haz clic en un dispositivo descubierto para rellenar previamente el asistente, o elige un punto de
partida manualmente como se describe a continuación.

El asistente de configuración adapta los pasos que muestra según tus elecciones:

- Elegir un **perfil integrado** rellena previamente el controlador, el área de trabajo y el cabezal
  — el asistente salta directamente a los pasos de rotativo, cámara y revisión
- **Importar un perfil** conserva los pasos de hardware y cabezal para que puedas corregir cualquier
  cosa que la importación haya interpretado mal
- **Device Not Listed** te guía a través de cada paso, incluidos los pasos de controlador y de
  consulta de especificaciones con IA

### Elegir un Punto de Partida

Elige un perfil de dispositivo integrado para rellenar previamente el controlador, el área de
trabajo y los ajustes de cabezal, o haz clic en **Device Not Listed** para configurar todo
manualmente. También puedes **Import from File…** un perfil exportado previamente o un perfil de
dispositivo LightBurn (.lbdev) con calibración de cámara y ajustes de láser.

![Asistente — Elegir un Punto de Partida](/screenshots/config-wizard-profile.webp)

### Elegir un Controlador

Elige la familia de firmware o protocolo que coincida con la placa controladora de tu máquina (GRBL,
Marlin, Smoothie, Ruida, OctoPrint, …). Elige **None — G-code export only** si solo quieres exportar
G-code a archivos y nunca manejar una máquina física. Este paso se omite cuando empiezas desde un
perfil integrado o una importación.

![Asistente — Elegir un Controlador](/screenshots/config-wizard-controller.webp)

### Conexión

Ingresa los parámetros de conexión que tu máquina requiere. Los campos exactos dependen del
controlador que elegiste:

- **Controladores serie** — ruta del dispositivo USB (p. ej. `/dev/ttyUSB0` en Linux, `COM3` en
  Windows) y velocidad de transmisión
- **Controladores de red** — dirección del host y puerto (p. ej. `192.168.1.100`)
- **OctoPrint** — URL del servidor y clave API

![Asistente — Conexión](/screenshots/config-wizard-connect.webp)

### Descubrir el Dispositivo

Cuando tu controlador lo soporta, el asistente ofrece conectarse al dispositivo y leer su
configuración automáticamente — área de trabajo, velocidades, aceleración y capacidades del
firmware. Esto funciona por serie USB **y por la red** (descubrimiento mDNS para OctoPrint y ESP3D).
Haz clic en **Probe Now** para detectar automáticamente estos valores, o usa **Next** para
ingresarlos manualmente en los siguientes pasos.

![Asistente — Descubrir el Dispositivo](/screenshots/config-wizard-probe.webp)

### Proveedor de IA

Se muestra solo cuando todavía no se ha configurado un proveedor de IA. Ingresa un endpoint
compatible con OpenAI (URL base y clave API) para que el siguiente paso pueda consultar las
especificaciones de máquinas comerciales conocidas. Omite este paso para ingresar los valores
manualmente.

![Asistente — Proveedor de IA](/screenshots/config-wizard-ai-provider.webp)

### Consulta de Especificaciones con IA

Si tu máquina es un modelo comercial conocido, la IA puede rellenar previamente sus especificaciones
a partir de la documentación del fabricante. Ingresa el fabricante y el modelo, luego haz clic en
**Look Up Specs**. Los valores sugeridos aparecen como filas de interruptores y empiezan aceptados —
desactiva cualquier cosa que no quieras aplicar.

![Asistente — Consulta de Especificaciones con IA](/screenshots/config-wizard-ai-lookup.webp)

### Hardware

Configura el montaje físico de la máquina:

- **Ejes** — extensiones del área de trabajo X/Y y la esquina de origen de coordenadas (0,0)
- **Dirección del eje** — invierte un eje si las coordenadas salen negativas
- **Eje Z** — si la máquina tiene un eje Z (motor de enfoque, cama móvil); cuando está ausente, no
  se generan movimientos Z y el lienzo 3D coloca el contenido en el plano de grabado
- **Orientación del panel** — rota el espacio de trabajo plano tal como se presenta en pantalla
  (Nativo, Girar a la izquierda, Girar a la derecha); las capas rotativas requieren Nativo
- **Área de trabajo** — márgenes alrededor del espacio inutilizable de la superficie de trabajo
- **Límites suaves** — límites de seguridad opcionales para el desplazamiento
- **Velocidades** — velocidad máxima de desplazamiento, velocidad máxima de corte y aceleración
- **Comportamiento** — ir al origen al inicio y homing de un solo eje

![Asistente — Hardware](/screenshots/config-wizard-hardware.webp)

### Cabezal

Declara qué está montado en el pórtico — un cabezal láser o un cabezal de husillo — y establece sus
parámetros. Para un láser: potencia máxima (valor S), tamaño del punto, frecuencia PWM y distancia
focal. Para un husillo: RPM máximas y mínimas.

![Asistente — Cabezal](/screenshots/config-wizard-head.webp)

### Módulo Rotativo

Configura opcionalmente un accesorio rotativo: tipo (mandril o rodillos), eje (A/B/C), modo (eje
real vs. reemplazo de eje), geometría y el indicador de dirección invertida. Omite este paso para
añadir un módulo rotativo más tarde desde la configuración de la máquina.

![Asistente — Módulo Rotativo](/screenshots/config-wizard-rotary.webp)

### Cámaras

Activa opcionalmente cualquier cámara que quieras usar para previsualización y alineación. Cuando
activas una cámara y continúas, se abre el
[asistente de cámara](../machine/camera.md#paso-2-asistente-de-cámara) para guiarte a través de la
configuración de imagen, la calibración de lente y la alineación de imagen. Puedes omitir esto y
configurar cámaras más tarde desde los ajustes de cámara de la máquina.

![Asistente — Cámaras](/screenshots/config-wizard-camera.webp)

### Revisión y Nombre

Dale un nombre a la máquina y revisa un resumen de todo lo que has configurado — controlador,
conexión, área de trabajo, velocidades, cabezales, módulos rotativos y cámaras. El asistente también
muestra advertencias, como un controlador faltante o un área de trabajo sin establecer.

![Asistente — Revisión y Nombre](/screenshots/config-wizard-review.webp)

Haz clic en **Create Machine** para finalizar. Se abre el diálogo de Configuración de Máquina para
tu nueva máquina, donde puedes ajustar cualquiera de los ajustes que el asistente rellenó
previamente. Consulta las páginas de [Configuración de Máquina](../machine/general.md) para más
detalles.

## Paso 3: Conexión Automática

Rayforge se conecta automáticamente a tu máquina cuando se inicia la aplicación (si la máquina está
encendida y conectada). No necesitas hacer clic manualmente en un botón de conexión.

El estado de conexión se muestra en la esquina inferior izquierda de la ventana principal con un
ícono de estado y una etiqueta que muestra el estado actual (Conectado, Conectando, Desconectado,
Error, etc.).

:::success ¡Conectado! Si tu máquina muestra el estado "Conectado", ¡estás listo para empezar a usar
Rayforge! :::

---

## Solución de Problemas de Conexión

### Dispositivo No Encontrado

- **Linux (Serial)**: Añade tu usuario al grupo `dialout`. Esto es requerido para **instalaciones
  Snap y no Snap** en distribuciones basadas en Debian para evitar mensajes AppArmor DENIED:

  ```bash
  sudo usermod -a -G dialout $USER
  ```

  Cierra sesión y vuelve a entrar para que los cambios surtan efecto.

- **Paquete Snap**: Además del grupo `dialout` arriba, asegúrate de haber otorgado permisos de
  puerto serie:

  ```bash
  sudo snap connect rayforge:serial-port
  ```

- **Windows**: Revisa el Administrador de Dispositivos para confirmar que el dispositivo es
  reconocido y anota el número de puerto COM.

### Conexión Rechazada

- Verifica que la dirección IP y el número de puerto sean correctos
- Asegúrate de que tu máquina esté encendida y conectada a la red
- Revisa la configuración del firewall si usas conexión de red

### La Máquina No Responde

- Prueba con una velocidad de transmisión diferente (algunos dispositivos usan `9600` o `57600`)
- Revisa si hay cables sueltos o conexiones deficientes
- Apaga y enciende tu cortador láser y vuelve a intentarlo

Para más ayuda, ver [Problemas de Conexión](../troubleshooting/connection.md).

---

**Siguiente:** [Guía de Inicio Rápido →](quick-start)
