# Modo Proyector

El Modo Proyector muestra tu área de corte en una ventana separada, diseñada
para mostrarse en un proyector externo o monitor secundario. Esto te permite
ver exactamente dónde cortará el láser proyectando las trayectorias
directamente sobre tu material, facilitando la alineación.

La ventana del proyector muestra tus workpieces renderizados en verde brillante
sobre un fondo negro. Muestra el marco de extensión de ejes de la máquina y el
origen de trabajo para que puedas ver el área de corte completa y dónde está el
punto de origen. La vista se actualiza en tiempo real mientras mueves o
modificas workpieces en el lienzo principal.

## Abrir la ventana del proyector

Abre la ventana del proyector desde **Ver - Mostrar diálogo del proyector**.
La ventana se abre como una ventana separada e independiente que puedes
arrastrar a cualquier pantalla conectada a tu sistema.

Un interruptor controla la ventana del proyector — el mismo elemento del menú
la cierra, y presionar Escape mientras la ventana del proyector está enfocada
también la cierra.

## Modo de pantalla completa

Haz clic en el botón **Pantalla completa** en la barra de título de la ventana
del proyector para entrar en modo de pantalla completa. Esto oculta las
decoraciones de la ventana y rellena toda la pantalla. Haz clic en **Salir de
pantalla completa** (el mismo botón) para volver al modo de ventana.

La pantalla completa es el modo previsto al proyectar sobre material, ya que
elimina el marco de ventana distractor y utiliza toda la superficie de la
pantalla.

## Opacidad

El botón de opacidad en la barra de título recorre cuatro niveles: 100%, 80%,
60% y 40%. Reducir la opacidad hace que la ventana del proyector sea
semitransparente, lo que puede ser útil en un monitor de escritorio para ver
las ventanas detrás de ella. Cada clic avanza al siguiente nivel de opacidad y
vuelve al inicio.

![Modo Proyector](/screenshots/addon-projector-mode.png)

## Qué muestra el proyector

La pantalla del proyector renderiza una vista simplificada de tu documento. Los
workpieces aparecen como contornos verdes brillantes que muestran las
trayectorias calculadas — las mismas rutas que se enviarán al láser. Las
imágenes base de tus workpieces no se muestran, manteniendo la pantalla
enfocada en las trayectorias de corte.

El marco de extensión de la máquina aparece como un borde que representa toda
el área de recorrido de los ejes de tu máquina. La cruz del origen de trabajo
muestra dónde se encuentra el origen del sistema de coordenadas dentro de esa
área. Ambos se actualizan automáticamente si cambias el desplazamiento del
sistema de coordenadas de trabajo en tu máquina.

## Temas relacionados

- [Sistemas de coordenadas](../general-info/coordinate-systems) - Entender las coordenadas de máquina y los desplazamientos de trabajo
- [Posicionamiento de workpieces](../features/workpiece-positioning) - Posicionar workpieces en el lienzo
