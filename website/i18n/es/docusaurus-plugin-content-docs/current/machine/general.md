---
description:
  "Configura los ajustes generales de la máquina en Rayforge: establece el nombre, selecciona un
  controlador y configura velocidades y aceleración."
---

# Ajustes generales

La página General en los Ajustes de máquina contiene el nombre de la máquina, la selección del
controlador y los ajustes de conexión, así como los parámetros de velocidad.

![Ajustes generales](/screenshots/machine-settings-general.webp)

## Nombre de la máquina

Dale a tu máquina un nombre descriptivo. Esto ayuda a identificarla en el desplegable de selección
cuando tienes varias máquinas configuradas.

## Controlador

Selecciona el controlador que corresponda al control de tu máquina. El controlador gestiona la
comunicación entre Rayforge y el hardware.

Los dispositivos GRBL tienen tres opciones de controlador serie:

- **GRBL (Serial)** — Controlador con contador de búfer, detección de bloqueos y recuperación de
  paradas. Recomendado para la mayoría de dispositivos GRBL
- **GRBL (Serial Simple)** — Controlador de protocolo ping-pong. Envía una línea, espera "ok", envía
  la siguiente. Sin gestión de búfer ni detección de bloqueos. Útil cuando el controlador estándar
  genera falsas alarmas
- **GRBL (Rust)** — Controlador experimental cuya pila completa del protocolo serie GRBL (control de
  flujo, transmisión de trabajos, detección de paradas, recuperación de bloqueos, ajustes y probing)
  se ejecuta en Rust. Puede seleccionarse como alternativa directa a GRBL (Serial)

Los controladores basados en Ruida son compatibles con el controlador **Ruida RPA**, que se conecta
directamente por USB o UDP, o vía TUI RPC a través del Ruida Protocol Analyzer.

### Vinculación del Puerto Serie

En lugar de una ruta de dispositivo (p. ej. `/dev/ttyUSB0` o `COM3`), el campo del puerto serie
también acepta un identificador USB `VID:PID` como `0403:6001`. Cuando una máquina se vincula por
VID:PID, la reconexión automática la sigue a su nuevo puerto después de que el sistema operativo
reenumere los dispositivos USB — por ejemplo tras un reinicio o al desenchufar y volver a enchufar.
Puedes encontrar el VID:PID de un dispositivo en la salida de `lsusb` (Linux) o en el Administrador
de dispositivos → Identificadores de hardware (Windows).

Tras seleccionar un controlador, aparecerán debajo del selector los ajustes específicos de conexión
(p. ej., puerto serie, baud rate). Estos varían según el controlador elegido.

<!-- prettier-ignore-start -->
:::tip
Un banner de error en la parte superior de la página te avisa si el controlador no está
configurado o si encuentra un problema.
:::
<!-- prettier-ignore-end -->

## Velocidades y aceleración

Estos ajustes controlan las velocidades máximas y la aceleración. Se usan para la estimación del
tiempo de trabajo y la optimización de trayectorias.

### Velocidad máxima de desplazamiento

La velocidad máxima para movimientos rápidos (sin corte) cuando el láser está apagado y el cabezal
se mueve a una nueva posición.

- **Rango típico**: 2000–5000 mm/min
- **Nota**: La velocidad real también está limitada por los ajustes de tu firmware. Este campo está
  deshabilitado si el dialecto de G-code seleccionado no permite especificar una velocidad de
  desplazamiento.

### Velocidad máxima de corte

La velocidad máxima permitida durante las operaciones de corte o grabado.

- **Rango típico**: 500–2000 mm/min
- **Nota**: Operaciones individuales pueden usar velocidades inferiores

### Aceleración

La tasa a la que la máquina acelera y desacelera, usada para estimaciones de tiempo y para calcular
la distancia de overscan predeterminada.

- **Rango típico**: 500–2000 mm/s²
- **Nota**: Debe coincidir o ser inferior a los ajustes de aceleración del firmware

<!-- prettier-ignore-start -->
:::tip
Comienza con valores de velocidad conservadores y auméntalos gradualmente. Observa tu máquina
para detectar saltos de correa, bloqueos de motor o pérdida de precisión de posicionamiento.
:::
<!-- prettier-ignore-end -->

## Exportar un perfil de máquina

Haz clic en el icono de compartir en la barra de encabezado del diálogo de ajustes para exportar la
configuración actual de la máquina. Elige una carpeta para guardar. Se creará un archivo ZIP con los
ajustes de la máquina y su dialecto de G-code, que puedes compartir con otros usuarios o importar en
otro sistema.

## Ver también

- [Configuración Inicial](../getting-started/first-time-setup.md) - Crea una máquina paso a paso con
  el asistente de configuración
- [Ajustes de hardware](hardware) - Dimensiones del área de trabajo y configuración de ejes
- [Ajustes de dispositivo](device) - Leer y escribir ajustes del firmware en el controlador
