---
description: "Usa el asistente de configuración para detectar y configurar automáticamente un dispositivo conectado consultando sus ajustes de firmware."
---

# Asistente de Configuración

El asistente de configuración detecta automáticamente tu dispositivo
conectándose a él y leyendo sus ajustes de firmware. Crea un perfil de
máquina completamente configurado en segundos, eliminando la configuración
manual.

## Iniciar el Asistente

1. Abre **Configuración → Máquinas** y haz clic en **Add Machine**
2. En el selector de perfiles, haz clic en **Other Device…** en la parte
   inferior

Esto abre el asistente. **No** requiere un perfil de dispositivo
existente — el asistente crea uno desde cero consultando el hardware
conectado.

## Conectar

La primera página te pide seleccionar un controlador y proporcionar los
parámetros de conexión.

![Asistente Conectar](/screenshots/app-settings-machines-wizard-connect.png)

### Selección de Controlador

Elige el controlador que coincida con el controlador de tu dispositivo.
Solo se muestran los controladores que soportan la detección.
Típicamente:

- **GRBL (Serial)** — Dispositivos GRBL conectados por USB
- **GRBL (Network)** — Dispositivos GRBL WiFi/Ethernet

### Parámetros de Conexión

Después de seleccionar un controlador, completa los detalles de conexión
(puerto serie, baudios, host, etc.). Estos son los mismos parámetros que en
los [Ajustes Generales](general).

Haz clic en **Next** para iniciar la detección.

## Descubrir

El asistente se conecta al dispositivo y consulta su firmware para obtener
los datos de configuración. Esto incluye:

- Versión del firmware e información de compilación (`$I`)
- Todos los ajustes del firmware (`$$`)
- Recorridos de ejes, velocidades, aceleración y rango de potencia del láser

Este paso suele completarse en unos segundos.

## Revisar

Después de una detección exitosa, la página de revisión muestra todos los
ajustes descubiertos. Todo está pre-rellenado pero se puede ajustar antes
de crear la máquina.

![Asistente Revisar](/screenshots/app-settings-machines-wizard-review.png)

### Información del Dispositivo

Información de solo lectura detectada desde el firmware:

- **Nombre del dispositivo** — extraído de la información de compilación
  del firmware
- **Versión del firmware** — p.ej. `1.1h`
- **Tamaño del búfer RX** — tamaño del búfer de recepción serie
- **Tolerancia de arco** — tolerancia de interpolación de arcos del firmware

### Área de Trabajo

- **Recorrido X** / **Recorrido Y** — recorrido máximo de los ejes en
  unidades de máquina, leído de los ajustes de firmware `$130` y `$131`

### Velocidad

- **Velocidad máx. de desplazamiento** — el menor valor entre `$110` y `$111`
- **Velocidad máx. de corte** — por defecto igual a la de desplazamiento;
  ajústala según sea necesario

### Aceleración

- **Aceleración** — el menor valor entre `$120` y `$121`, en unidades de
  máquina por segundo al cuadrado

### Láser

- **Potencia máx. (valor S)** — del ajuste de firmware `$30` (spindle máx.)

### Comportamiento

- **Home al iniciar** — activado si el homing del firmware (`$22`) está
  activado
- **Homing mono-eje** — activado si el firmware fue compilado con el flag `H`

### Advertencias

El asistente puede mostrar advertencias sobre problemas potenciales, como:

- El modo láser no está activado (`$32=0`)
- El dispositivo informa en pulgadas (`$13=1`)

## Crear la Máquina

Haz clic en **Create Machine** para finalizar. El asistente emite el perfil
configurado y el proceso normal de creación de máquina continúa — el
[diálogo de ajustes de máquina](general) se abre para realizar ajustes
adicionales.

## Limitaciones

- El asistente solo funciona con controladores que soportan la detección.
  Si tu controlador no está listado, usa un perfil predefinido del selector.
- La detección requiere que el dispositivo esté encendido y conectado.
- Algunos ajustes del firmware pueden no ser legibles en todos los
  dispositivos.

## Ver también

- [Ajustes Generales](general) — configuración manual de la máquina
- [Ajustes de dispositivo](device) — leer y escribir ajustes del firmware
  en una máquina ya configurada
- [Añadir una Máquina](../application-settings/machines) — descripción
  general del proceso de creación de máquinas
