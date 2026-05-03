---
description: "Configura los ajustes generales de la máquina en Rayforge: establece el nombre, selecciona un controlador y configura velocidades y aceleración."
---

# Ajustes generales

La página General en los Ajustes de máquina contiene el nombre de la
máquina, la selección del controlador y los ajustes de conexión, así como
los parámetros de velocidad.

![Ajustes generales](/screenshots/machine-general.png)

## Nombre de la máquina

Dale a tu máquina un nombre descriptivo. Esto ayuda a identificarla en el
desplegable de selección cuando tienes varias máquinas configuradas.

## Controlador

Selecciona el controlador que corresponda al control de tu máquina. El
controlador gestiona la comunicación entre Rayforge y el hardware.

Tras seleccionar un controlador, aparecerán debajo del selector los ajustes
específicos de conexión (p. ej., puerto serie, baud rate). Estos varían según
el controlador elegido.

:::tip
Un banner de error en la parte superior de la página te avisa si el
controlador no está configurado o si encuentra un problema.
:::

## Velocidades y aceleración

Estos ajustes controlan las velocidades máximas y la aceleración. Se usan
para la estimación del tiempo de trabajo y la optimización de trayectorias.

### Velocidad máxima de desplazamiento

La velocidad máxima para movimientos rápidos (sin corte) cuando el láser
está apagado y el cabezal se mueve a una nueva posición.

- **Rango típico**: 2000–5000 mm/min
- **Nota**: La velocidad real también está limitada por los ajustes de tu
  firmware. Este campo está deshabilitado si el dialecto de G-code
  seleccionado no permite especificar una velocidad de desplazamiento.

### Velocidad máxima de corte

La velocidad máxima permitida durante las operaciones de corte o grabado.

- **Rango típico**: 500–2000 mm/min
- **Nota**: Operaciones individuales pueden usar velocidades inferiores

### Aceleración

La tasa a la que la máquina acelera y desacelera, usada para estimaciones
de tiempo y para calcular la distancia de overscan predeterminada.

- **Rango típico**: 500–2000 mm/s²
- **Nota**: Debe coincidir o ser inferior a los ajustes de aceleración del
  firmware

:::tip
Comienza con valores de velocidad conservadores y auméntalos gradualmente.
Observa tu máquina para detectar saltos de correa, bloqueos de motor o
pérdida de precisión de posicionamiento.
:::

## Exportar un perfil de máquina

Haz clic en el icono de compartir en la barra de encabezado del diálogo de
ajustes para exportar la configuración actual de la máquina. Elige una carpeta
para guardar. Se creará un archivo ZIP con los ajustes de la máquina y su
dialecto de G-code, que puedes compartir con otros usuarios o importar en otro
sistema.

## Ver también

- [Asistente de Configuración](config-wizard) - Detectar y configurar
  automáticamente un dispositivo conectado
- [Ajustes de hardware](hardware) - Dimensiones del área de trabajo y
  configuración de ejes
- [Ajustes de dispositivo](device) - Leer y escribir ajustes del firmware en
  el controlador
