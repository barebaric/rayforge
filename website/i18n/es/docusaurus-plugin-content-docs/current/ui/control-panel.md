# Panel de Control

El Panel de Control en la parte inferior de la ventana de Rayforge proporciona control manual sobre la posición de tu cortador láser, estado de la máquina en tiempo real y una vista de registro para monitorear operaciones.

## Resumen

El Panel de Control combina varias funciones en una interfaz conveniente:

1. **Controles de Desplazamiento**: Movimiento y posicionamiento manual
2. **Estado de la Máquina**: Posición en tiempo real y estado de conexión
3. **Consola**: Terminal de código G interactiva con resaltado de sintaxis
4. **Sistema de Coordenadas de Trabajo (WCS)**: Selección rápida de WCS

![Panel de Control](/screenshots/control-panel.png)

## Accediendo al Panel de Control

El Panel de Control siempre está visible en la parte inferior de la ventana principal. Puede
alternarse vía:

- **Menú**: Ver → Panel de Control
- **Atajo de Teclado**: Ctrl+L

:::note Se Requiere Conexión
Los controles de desplazamiento solo están disponibles cuando se está conectado a una máquina
que soporta operaciones de desplazamiento.
:::


## Controles de Desplazamiento

Los controles de desplazamiento proporcionan control manual sobre la posición de tu cortador láser,
permitiéndote mover precisamente la cabeza del láser para configuración, alineación y
propósitos de prueba.

### Controles de Homing

Lleva los ejes de tu máquina al origen para establecer una posición de referencia:

| Botón    | Función        | Descripción                            |
| -------- | -------------- | -------------------------------------- |
| Home X   | Origen eje X  | Mueve el eje X a su posición de origen |
| Home Y   | Origen eje Y  | Mueve el eje Y a su posición de origen |
| Home Z   | Origen eje Z  | Mueve el eje Z a su posición de origen |
| Home Todo| Origen todos los ejes | Lleva todos los ejes al origen simultáneamente |

:::tip Secuencia de Homing
Se recomienda llevar todos los ejes al origen antes de iniciar cualquier trabajo para asegurar
posicionamiento preciso.
:::


### Movimiento Direccional

Los controles de desplazamiento proporcionan botones para movimiento direccional:

```
  ↖  ↑  ↗
  ←  •  →
  ↙  ↓  ↘
```

| Botón            | Movimiento                       | Atajo de Teclado |
| ---------------- | -------------------------------- | ----------------- |
| ↑                | Y+ (Y- si la máquina tiene Y invertido) | Flecha Arriba    |
| ↓                | Y- (Y+ si la máquina tiene Y invertido) | Flecha Abajo     |
| ←                | X- (izquierda)                   | Flecha Izquierda  |
| →                | X+ (derecha)                     | Flecha Derecha    |
| ↖ (superior-izquierda) | X- Y+/- (diagonal)        | -                 |
| ↗ (superior-derecha) | X+ Y+/- (diagonal)          | -                 |
| ↙ (inferior-izquierda) | X- Y-/+ (diagonal)        | -                 |
| ↘ (inferior-derecha) | X+ Y-/+ (diagonal)          | -                 |
| Z+               | Eje Z hacia arriba               | Re Pág            |
| Z-               | Eje Z hacia abajo                | Av Pág            |

:::note Se Requiere Foco
Los atajos de teclado solo funcionan cuando la ventana principal tiene foco.
:::


### Retroalimentación Visual

Los botones de desplazamiento proporcionan retroalimentación visual:

- **Normal**: El botón está habilitado y es seguro de usar
- **Advertencia (naranja)**: El movimiento se acercaría o excedería los límites suaves
- **Deshabilitado**: El movimiento no está soportado o la máquina no está conectada

### Ajustes de Desplazamiento

Configura el comportamiento de las operaciones de desplazamiento:

**Velocidad de Desplazamiento:**
- **Rango**: 1-10,000 mm/min
- **Por defecto**: 1,000 mm/min
- **Propósito**: Controla qué tan rápido se mueve la cabeza del láser

:::tip Selección de Velocidad
- Usa velocidades más bajas (100-500 mm/min) para posicionamiento preciso
- Usa velocidades más altas (1,000-3,000 mm/min) para movimientos grandes
- Velocidades muy altas pueden causar pasos perdidos en algunas máquinas
:::


**Distancia de Desplazamiento:**
- **Rango**: 0.1-1,000 mm
- **Por defecto**: 10.0 mm
- **Propósito**: Controla qué tan lejos se mueve la cabeza del láser por presión de botón

:::tip Selección de Distancia
- Usa distancias pequeñas (0.1-1.0 mm) para ajuste fino
- Usa distancias medianas (5-20 mm) para posicionamiento general
- Usa distancias grandes (50-100 mm) para reposicionamiento rápido
:::


## Pantalla de Estado de la Máquina

El Panel de Control muestra información en tiempo real sobre tu máquina:

### Posición Actual

Muestra la posición de la cabeza del láser en el sistema de coordenadas activo:

- Las coordenadas son relativas al origen WCS seleccionado
- Se actualiza en tiempo real mientras te desplazas o ejecutas trabajos
- Formato: Valores X, Y, Z en milímetros

### Estado de Conexión

- **Conectado**: Indicador verde, la máquina está respondiendo
- **Desconectado**: Indicador gris, sin conexión a máquina
- **Error**: Indicador rojo, problema de conexión o comunicación

### Estado de la Máquina

- **Idle**: La máquina está lista para comandos
- **Run**: Trabajo actualmente en ejecución
- **Hold**: Trabajo pausado
- **Alarm**: La máquina está en estado de alarma
- **Home**: Ciclo de homing en progreso

## Sistema de Coordenadas de Trabajo (WCS)

El Panel de Control proporciona acceso rápido a la gestión del Sistema de Coordenadas de Trabajo.

### Selección del Sistema Activo

Selecciona qué sistema de coordenadas está actualmente activo:

| Opción         | Tipo  | Descripción                                     |
| -------------- | ----- | ----------------------------------------------- |
| G53 (Máquina)  | Fijo  | Coordenadas absolutas de máquina, no se pueden cambiar |
| G54 (Trabajo 1)| Usuario | Primer sistema de coordenadas de trabajo        |
| G55 (Trabajo 2)| Usuario | Segundo sistema de coordenadas de trabajo       |
| G56 (Trabajo 3)| Usuario | Tercer sistema de coordenadas de trabajo        |
| G57 (Trabajo 4)| Usuario | Cuarto sistema de coordenadas de trabajo        |
| G58 (Trabajo 5)| Usuario | Quinto sistema de coordenadas de trabajo        |
| G59 (Trabajo 6)| Usuario | Sexto sistema de coordenadas de trabajo         |

### Desplazamientos Actuales

Muestra los valores de desplazamiento para el WCS activo:

- Mostrados como (X, Y, Z) en milímetros
- Representan la distancia desde el origen de máquina al origen WCS
- Se actualiza automáticamente cuando cambian los desplazamientos WCS

### Estableciendo Cero WCS

Define dónde debería estar el origen del WCS activo:

| Botón  | Función | Descripción                                          |
| ------ | ------- | ---------------------------------------------------- |
| Cero X | Establecer X=0 | Hace que la posición X actual sea el origen X para el WCS activo |
| Cero Y | Establecer Y=0 | Hace que la posición Y actual sea el origen Y para el WCS activo |
| Cero Z | Establecer Z=0 | Hace que la posición Z actual sea el origen Z para el WCS activo |

:::note G53 No Puede Cambiarse
Los botones de cero están deshabilitados cuando G53 (Coordenadas de Máquina) está seleccionado,
ya que las coordenadas de máquina están fijadas por hardware.
:::


:::tip Flujo de Trabajo para Establecer WCS
1. Conéctate a tu máquina y lleva todos los ejes al origen
2. Selecciona el WCS que quieres configurar (ej., G54)
3. Desplaza la cabeza del láser a la posición de origen deseada
4. Haz clic en Cero X y Cero Y para establecer esta posición como (0, 0)
5. El desplazamiento se almacena en el controlador de tu máquina
:::


## Consola

La Consola proporciona una interfaz estilo terminal interactiva para enviar comandos de código G
y monitorear la comunicación de la máquina:

### Entrada de Comandos

La caja de entrada de comandos te permite enviar código G crudo directamente a la máquina:

- **Soporte Multi-línea**: Pega o escribe múltiples comandos
- **Enter**: Envía todos los comandos
- **Shift+Enter**: Inserta una nueva línea (para editar antes de enviar)
- **Historial**: Usa flechas Arriba/Abajo para navegar comandos enviados previamente

### Pantalla de Registro

El registro muestra la comunicación entre Rayforge y tu máquina con
resaltado de sintaxis para fácil lectura:

- **Comandos de Usuario** (azul): Comandos que ingresaste o enviaste durante trabajos
- **Marcas de Tiempo** (gris): Hora de cada mensaje
- **Errores** (rojo): Mensajes de error de la máquina
- **Advertencias** (naranja): Mensajes de advertencia
- **Sondeos de Estado** (tenue): Reportes de posición/estado en tiempo real como
  `<Idle|WPos:0.000,0.000,0.000|...>`

### Modo Verbose

Haz clic en el ícono de terminal en la esquina superior derecha de la consola para alternar
la salida verbose:

- **Apagado** (por defecto): Oculta sondeos de estado frecuentes y respuestas "ok"
- **Encendido**: Muestra toda la comunicación de la máquina

### Comportamiento de Auto-Desplazamiento

La consola se desplaza automáticamente para mostrar nuevos mensajes:

- Desplazarse hacia arriba deshabilita el auto-desplazamiento para que puedas revisar el historial
- Desplazarse al final rehabilita el auto-desplazamiento
- Los nuevos mensajes aparecen inmediatamente cuando el auto-desplazamiento está activo

### Usando la Consola para Solución de Problemas

La consola es invaluable para diagnosticar problemas:

- Verifica que los comandos se estén enviando correctamente
- Revisa mensajes de error del controlador
- Monitorea el estado de conexión y estabilidad
- Revisa el progreso de ejecución del trabajo en tiempo real
- Envía comandos de diagnóstico (ej., `$$` para ver ajustes de GRBL)

## Compatibilidad de Máquina

El Panel de Control se adapta a las capacidades de tu máquina:

### Soporte de Ejes

- **Eje X/Y**: Soportado por prácticamente todos los cortadores láser
- **Eje Z**: Solo disponible en máquinas con control de eje Z
- **Movimiento Diagonal**: Requiere soporte para ambos ejes X e Y

### Tipos de Máquina

| Tipo de Máquina       | Soporte de Desplazamiento | Notas                     |
| --------------------- | ------------------------- | ------------------------- |
| GRBL (v1.1+)          | Completo                  | Soporta todas las funciones de desplazamiento |
| Smoothieware          | Completo                  | Soporta todas las funciones de desplazamiento |
| Controladores Personalizados | Variable    | Depende de la implementación |

## Funciones de Seguridad

### Límites Suaves

Cuando los límites suaves están habilitados en tu perfil de máquina:

- Los botones muestran advertencia naranja al acercarse a los límites
- El movimiento se limita automáticamente para prevenir exceder los límites
- Proporciona retroalimentación visual para prevenir choques

### Estado de Conexión

- Todos los controles están deshabilitados cuando no está conectado a una máquina
- Los botones actualizan la sensibilidad según el estado de la máquina
- Previene movimiento accidental durante la operación

---

**Páginas Relacionadas:**

- [Sistemas de Coordenadas de Trabajo (WCS)](../general-info/work-coordinate-systems) - Gestionando WCS
- [Configuración de Máquina](../machine/general) - Configura tu máquina
- [Atajos de Teclado](../reference/shortcuts) - Referencia completa de atajos
- [Ventana Principal](main-window) - Resumen de la interfaz principal
- [Ajustes Generales](../machine/general) - Configuración del dispositivo
