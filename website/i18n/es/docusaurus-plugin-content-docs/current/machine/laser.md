# Ajustes de Láser

La página de Láser en Configuración de Máquina configura tu(s) cabezal(es) láser y sus propiedades.

![Ajustes de Láser](/screenshots/machine-settings-laser.webp)

## Cabezales Láser

Rayforge soporta máquinas con múltiples cabezales láser. Cada cabezal láser tiene su propia
configuración.

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

El nivel de potencia usado para operaciones de enmarcado (delimitando sin cortar).

- Establecer en 0 para deshabilitar el enmarcado
- Ajusta según tu láser y material

#### Velocidad de Enmarcado

La velocidad a la que se mueve el cabezal láser durante el enmarcado. Se establece por cabezal
láser, por lo que si tu máquina tiene varios láseres con diferentes características, puedes elegir
una velocidad adecuada para cada uno. Velocidades más lentas hacen que la ruta de enmarcado sea más
fácil de seguir visualmente.

#### Potencia de Enfoque

El nivel de potencia usado cuando el modo enfoque está activado. El modo enfoque enciende el láser a
baja potencia para actuar como "puntero láser" para posicionamiento.

- Establecer en 0 para deshabilitar la función de modo enfoque
- Usar para alineación visual y posicionamiento

<!-- prettier-ignore-start -->
:::tip[Usando el Modo Enfoque]
Haz clic en el botón de enfoque (icono de láser) en la barra de
herramientas para alternar el modo enfoque. El láser se encenderá a este nivel de potencia,
ayudándote a ver exactamente dónde está posicionado el láser. Consulta
[Posicionamiento de Piezas de Trabajo](../features/workpiece-positioning.md) para más información.
:::
<!-- prettier-ignore-end -->

#### Tamaño del Punto

El tamaño físico de tu haz láser enfocado en milímetros.

- Ingresa ambas dimensiones X e Y
- La mayoría de los láseres tienen un punto circular (ej., 0.1 x 0.1)
- Afecta los cálculos de calidad de grabado

<!-- prettier-ignore-start -->
:::tip[Midiendo el Tamaño del Punto]
Para medir el tamaño de tu punto:

1. Dispara un pulso corto a baja potencia en un material de prueba
2. Mide la marca resultante con calibradores
3. Usa el promedio de múltiples mediciones
:::
<!-- prettier-ignore-end -->

#### Color

El color usado para mostrar las operaciones de este láser (cortes y grabado) en el lienzo y la vista
previa 3D. Esto te ayuda a distinguir visualmente qué láser realizará cada operación cuando trabajas
con múltiples cabezales láser.

- Haz clic en la muestra de color para abrir un selector de color
- Elige un color que contraste bien con la vista previa de tu material
- Los colores predeterminados se asignan automáticamente

<!-- prettier-ignore-start -->
:::tip[Flujos de Trabajo Multi-Láser]
Al usar múltiples cabezales láser, asignar diferentes colores a
cada láser facilita ver qué operaciones serán realizadas por qué láser. Por ejemplo, usa rojo para
tu láser de corte principal y azul para un láser de grabado secundario.
:::
<!-- prettier-ignore-end -->

#### Tipo de Láser

Elige el tipo de cabezal láser en el menú desplegable:

- **Diodo**: Láseres de diodo estándar (los más comunes en máquinas de afición)
- **CO2**: Láseres de tubo CO2
- **Fibra**: Láseres de fibra

Al seleccionar CO2 o Fibra, aparecen **ajustes PWM** adicionales (ver más abajo). Para láseres de
diodo, la sección PWM se oculta ya que no aplica.

El tipo de láser también establece una **longitud de onda** predeterminada (utilizada por el modelo
de quemado físico) cuando no se introduce un valor explícito a continuación.

#### Longitud de onda (nm)

La longitud de onda de emisión de tu láser, en nanómetros. Alimenta el
[modelo de quemado físico](../ui/3d-preview.md#physical-burn-model) en la vista previa 3D: junto con
los datos de [absorción](../application-settings/materials.md#absorption) del material, determina
cuánta energía del láser absorbe el material de base.

Cuando se establece en 0, Rayforge usa la longitud de onda típica para el tipo de láser seleccionado
(ej. 445 nm para diodo, 1064 nm para fibra, 10600 nm para CO2).

#### Potencia Óptica Máxima (W)

La potencia de salida óptica de tu láser a potencia máxima, en vatios. Esta es la salida de luz
real, no la entrada eléctrica. Junto con el tamaño del punto y la velocidad de escaneo, determina la
fluencia (J/cm²) utilizada por el
[modelo de quemado físico](../ui/3d-preview.md#physical-burn-model).

Cuando se establece en 0, se usa un valor predeterminado de escritorio de rango medio.

#### Ajustes PWM

Cuando se selecciona un tipo de láser CO2 o Fibra, aparecen los siguientes controles PWM:

- **Frecuencia PWM**: La frecuencia PWM predeterminada en Hz para este cabezal láser. Los valores
  típicos van de 500 Hz a varios kHz dependiendo de tu controlador y fuente de alimentación.
- **Frecuencia PWM máxima**: El límite superior para el ajuste de frecuencia. Esto evita introducir
  valores que tu hardware no puede manejar.
- **Ancho de pulso**: El ancho de pulso predeterminado en microsegundos. Controla cuánto tiempo
  permanece encendido cada pulso durante un ciclo.
- **Ancho de pulso mín/máx**: Límites para el ajuste del ancho de pulso.

Estos valores predeterminados se transfieren a tus pasos de operación, donde pueden sobreescribirse
por paso si es necesario.

#### Desplazamiento del Puntero

Si tu máquina tiene un láser puntero separado (un pequeño láser de punto rojo) montado a una
distancia fija del haz de corte, puedes indicarle a Rayforge esa distancia para que la compense.

- **Usar desplazamiento del puntero**: activa la compensación. Desactivado por defecto.
- **Desplazamiento del puntero X / Y**: la distancia en milímetros del punto del haz de corte al
  punto del puntero, a lo largo de los ejes X e Y de la máquina.

Al activarlo cambian tres cosas:

1. **Establecer cero en la posición actual** (y los botones Cero X / Cero Y) coloca el origen de
   trabajo donde el _punto del puntero_ marca el material, no donde está el haz de corte
   (invisible).
2. El lienzo muestra un punto amarillo del puntero junto al punto rojo del haz, marcando dónde está
   el punto del puntero en tu material.
3. Un interruptor de **alineación del puntero** está disponible en el popover de movimiento (ver
   abajo).

#### Alineación del Puntero

La alineación del puntero es un interruptor de sesión en el popover de movimiento (el icono de
brújula junto a la lectura de posición). Mientras está activada, todas las operaciones absolutas de
apuntado — Mover a, los atajos de esquina, ir al origen del SCF, Clic para mover, Mover cabeza aquí
y enmarcar — se desplazan para que el _punto del puntero_ aterrice en la posición apuntada. El punto
del puntero en el lienzo se dibuja relleno mientras la alineación está activada y hueco mientras
está desactivada.

El flujo de trabajo típico:

1. Mueve la máquina hasta que el punto del puntero marque tu punto de referencia en el material.
2. **Establece el cero de trabajo** ahí — con el desplazamiento del puntero activado, el origen
   queda exactamente donde apuntó el puntero.
3. Activa la **alineación del puntero** en el popover de movimiento.
4. Enmarca y mueve con el punto del puntero: todo lo que apuntes queda marcado por el puntero.
5. Al pulsar **Enviar**, una advertencia te recuerda que el trabajo graba con el haz en las
   posiciones del SCF — puedes desactivar la alineación y grabar, grabar de todos modos o cancelar.

Dos cosas nunca se desplazan: el **jog** (un movimiento relativo no necesita compensación) y los
**trabajos** — el corte siempre ocurre con el haz en las posiciones del SCF, por lo que tu salida
G-code es idéntica tanto si la alineación está activada como desactivada. Junto con el zerado por el
punto del puntero, todo queda consistente: el origen está en `haz + desplazamiento`, el apuntado
desplaza cada objetivo `-desplazamiento`, y la grabación no se desplaza.

La alineación del puntero es una configuración de sesión: no se guarda en el perfil de la máquina y
se restablece al cambiar de máquina.

<!-- prettier-ignore-start -->
:::tip[Medir el Desplazamiento]
1. Mueve la máquina hasta que el punto del puntero marque un punto visible en el
   material.
2. Activa el modo enfoque y mueve hasta que el *haz de corte* queme una marca en
   exactamente el mismo punto (o mueve cuidadosamente el haz ahí con la potencia
   de enfoque).
3. El desplazamiento es la posición del puntero menos la del haz. Por ejemplo,
   si el puntero marcó X=100 y tuviste que mover el haz a X=88 para dar en el
   mismo punto, introduce X = 12.0 — el punto del puntero está 12 mm por
   delante del haz.

Si un corte de prueba sale desplazado, invierte el signo del eje
correspondiente.
:::
<!-- prettier-ignore-end -->

<!-- prettier-ignore-start -->
:::note[Modo Rotatorio]
Cuando el accesorio rotatorio está activo, el eje Y se reemplaza por el rodillo
rotatorio, por lo que la componente Y del desplazamiento del puntero no se
aplica de forma significativa. Ponla a 0 para trabajos rotatorios.
:::
<!-- prettier-ignore-end -->

#### Modelo 3D

Cada cabezal láser puede tener un modelo 3D asignado. Este modelo se renderiza en la
[vista 3D](../ui/3d-preview.md) y sigue la trayectoria durante la simulación.

Haz clic en la fila de selección de modelo para explorar los modelos disponibles. Una vez
seleccionado un modelo, puedes ajustar su escala, rotación (X/Y/Z) y distancia focal para coincidir
con tu cabezal láser físico.

## Ver También

- [Ajustes de Dispositivo](device) - Ajustes de modo láser de GRBL
- [Posicionamiento de Piezas de Trabajo](../features/workpiece-positioning.md) - Uso del modo
  enfoque y otros métodos de posicionamiento
