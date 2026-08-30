# Ajustes de Hardware

La página de Hardware en Configuración de Máquina configura las dimensiones físicas, sistema de
coordenadas y límites de movimiento de tu máquina.

![Ajustes de Hardware](/screenshots/machine-settings-hardware.webp)

## Ejes

Configura la extensión de los ejes y el sistema de coordenadas para tu máquina.

### Extensión X/Y

El rango completo de recorrido de cada eje en unidades de máquina.

- Mide el área de corte real, no el exterior de la máquina
- Ten en cuenta cualquier obstrucción o límite
- Ejemplo: 400 para un láser K40 típico

### Origen de Coordenadas

Selecciona dónde está ubicado el origen de coordenadas (0,0) de tu máquina. Esto determina cómo se
interpretan las coordenadas.

- **Abajo Izquierda**: Más común para dispositivos GRBL. X aumenta hacia la derecha, Y aumenta hacia
  arriba.
- **Arriba Izquierda**: Común para algunas máquinas estilo CNC. X aumenta hacia la derecha, Y
  aumenta hacia abajo.
- **Arriba Derecha**: X aumenta hacia la izquierda, Y aumenta hacia abajo.
- **Abajo Derecha**: X aumenta hacia la izquierda, Y aumenta hacia arriba.

#### Encontrando Tu Origen

1. Lleva tu máquina al origen usando el botón Home
2. Observa hacia dónde se mueve la cabeza del láser
3. Esa posición es tu origen (0,0)

:::info El ajuste del origen de coordenadas afecta cómo se genera el código G. Asegúrate de que
coincida con la configuración de homing de tu firmware. :::

### Dirección del Eje

Invierte la dirección de cualquier eje si es necesario:

- **Invertir Dirección del Eje X**: Hace que los valores de coordenadas X sean negativos
- **Invertir Dirección del Eje Y**: Hace que los valores de coordenadas Y sean negativos
- **Invertir Dirección del Eje Z**: Habilitar si un comando Z positivo (ej., G0 Z10) mueve la cabeza
  hacia abajo

### Eje Z

No todos los láseres tienen un eje Z — muchas máquinas de diodo simples de 2 ejes no lo tienen. Usa
estos ajustes para indicarle a Rayforge si tu máquina puede moverse en Z:

- **Tiene eje Z**: Activa si esta máquina tiene un eje Z (motor de enfoque, cama móvil o pórtico que
  puede moverse verticalmente). Cuando está desactivado, el eje Z se trata como ausente: no se
  generan movimientos Z, y el lienzo 3D coloca el contenido en el plano de grabado en lugar de
  mostrar una cabeza que se mueve a lo largo de Z.
- **Invertir Dirección del Eje Z**: Solo se muestra cuando **Tiene eje Z** está activado. Activa si
  un comando Z positivo (ej., G0 Z10) mueve la cabeza hacia abajo.

### Orientación del panel

La orientación del panel rota el espacio de trabajo plano tal como se presenta en pantalla y
reasigna la salida de vuelta a la máquina. Esto es útil cuando una cama orientada físicamente en
retrato es más fácil de editar en paisaje.

- **Nativo**: El espacio de trabajo se muestra exactamente como está orientada la máquina
- **Girar a la izquierda**: El espacio de trabajo se rota 90 grados en sentido antihorario
- **Girar a la derecha**: El espacio de trabajo se rota 90 grados en sentido horario

Las capas rotativas requieren la orientación **Nativo**.

## Área de Trabajo

Los márgenes definen el espacio inutilizable alrededor de los bordes de la extensión de tus ejes.
Esto es útil cuando tu máquina tiene áreas donde el láser no puede alcanzar (ej., debido al
ensamblaje de la cabeza del láser, cadenas de cables u otras obstrucciones).

- **Margen Izquierdo/Superior/Derecho/Inferior**: El espacio inutilizable desde cada borde en
  unidades de máquina

Cuando se establecen los márgenes, el área de trabajo (espacio utilizable) se calcula como la
extensión de los ejes menos los márgenes.

## Límites Suaves

Límites de seguridad configurables para desplazar la cabeza de la máquina. Cuando están habilitados,
los controles de desplazamiento impedirán el movimiento fuera de estos límites.

- **Habilitar Límites Suaves Personalizados**: Alternar para usar límites personalizados en lugar de
  los límites de la superficie de trabajo
- **X/Y Mín**: Coordenada mínima para cada eje
- **X/Y Máx**: Coordenada máxima para cada eje

Los límites suaves se restringen automáticamente para permanecer dentro de la extensión de los ejes
(0 al valor de extensión).

:::tip Usa límites suaves para proteger áreas de tu superficie de trabajo que nunca deben alcanzarse
durante el desplazamiento, como áreas con fijaciones o equipo sensible. :::

## Ver También

- [Ajustes Generales](general) - Nombre de máquina y ajustes de velocidad
- [Ajustes de Dispositivo](device) - Homing de GRBL y ajustes de ejes
