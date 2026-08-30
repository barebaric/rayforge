---
description: "El diseñador paramétrico 2D integrado de Rayforge le permite dibujar diseños basados en restricciones y guiados por dimensiones que permanecen editables y precisos."
---

# Diseñador paramétrico 2D

Rayforge incluye un diseñador paramétrico 2D para dibujar piezas directamente
en la aplicación. En lugar de importar ilustraciones terminadas de otro
programa, se dibujan líneas, curvas y formas en un lienzo infinito y se unen
entre sí con restricciones. El resultado es un diseño que permanece preciso sin
importar cuántas veces cambie de opinión sobre sus dimensiones.

![El editor de bocetos](/screenshots/addons-sketcher-editor.webp)

## Qué significa "paramétrico" aquí

Un boceto es más que un dibujo: es un pequeño modelo con reglas. Las reglas son
**restricciones**: afirmaciones como «estas dos líneas son paralelas», «esta
esquina es un ángulo recto» o «este borde mide exactamente 100 mm». Después de
cada cambio, un resolvedor reordena la geometría para que todas las reglas se
cumplan de nuevo.

Esto tiene una consecuencia práctica: puede capturar su intención de diseño
una sola vez y seguir editando después. Suba la restricción de distancia de
100 mm a 130 mm y toda la pieza seguirá el cambio. Las restricciones
dimensionales también aceptan expresiones: un radio de `width/2` sigue siendo
la mitad de la anchura, sea cual sea esta.

Cuando cada grado de libertad restante queda fijado por una restricción, el
boceto está *completamente restringido*. El editor le indica en qué punto está
mediante colores: la geometría sujeta por restricciones se dibuja en verde,
los puntos sin restringir en negro y, una vez que el boceto está completamente
restringido, el verde se oscurece. Las restricciones que se contradicen entre
sí se marcan en rojo y se enumeran en el panel de conflictos de la barra
lateral, donde puede inspeccionarlas o eliminarlas.

![Un boceto acotado](/screenshots/addons-sketcher-constraints.webp)

Un boceto subrestringido no es un error; a menudo es exactamente lo que se
quiere mientras se experimenta. La página [Restricciones](constraints.md)
explica en detalle cada tipo de restricción disponible.

## El editor de bocetos

Los bocetos viven en el documento como cualquier otra pieza de trabajo. Cree
uno con el botón **Nuevo boceto** del panel inferior (o haga clic derecho en
el lienzo y elija la misma entrada del menú contextual) y el editor de bocetos
toma el control de la ventana: el lienzo en el centro, un panel de propiedades
con el nombre del boceto y sus parámetros a la izquierda, y una barra de
herramientas arriba.

La barra de herramientas reúne las herramientas de nivel de sesión: deshacer y
rehacer, alternadores para la visibilidad de restricciones y geometría de
construcción, colores de relleno y línea, espejado, y los botones
**Finalizar** y **Cancelar**. **Finalizar** guarda el boceto de nuevo en el
documento; **Cancelar** descarta los cambios hechos en esta sesión. Para
volver a editar un boceto existente más tarde, haga doble clic en él en el
espacio de trabajo principal, o selecciónelo y elija **Editar boceto** en el
menú contextual.

El editor está pensado para el teclado. La barra de estado de la parte
inferior siempre muestra los atajos que se aplican a la herramienta y
selección actuales, de modo que las teclas relevantes están en pantalla justo
cuando las necesita. Hay deshacer y rehacer completos para cada operación.

## El menú circular

Hacer clic derecho en cualquier parte del editor de bocetos abre el menú
circular: un menú radial que pone cada herramienta de dibujo y modificación a
un clic de distancia. El menú es consciente del contexto: al hacer clic
derecho en el espacio vacío se ofrecen las herramientas de dibujo, mientras
que al hacerlo sobre una línea seleccionada se ofrecen las restricciones y
modificaciones que tienen sentido para una línea. Las herramientas
relacionadas se agrupan; pase el cursor sobre un grupo para desplegar sus
elementos. Haga clic derecho de nuevo para cerrar el menú o abrirlo en otro
lugar.

![El menú circular abierto sobre una línea seleccionada](/screenshots/addons-sketcher-pie-menu.webp)

## Atajos de teclado

El diseñador se opera desde el teclado, y la barra de estado de la parte
inferior siempre muestra los atajos que se aplican a la herramienta y
selección actuales. Estos atajos generales funcionan en todo el editor:

| Acción                                               | Atajo                                |
| ---------------------------------------------------- | ------------------------------------ |
| Herramienta de selección                             | `Space`                              |
| Deshacer / Rehacer                                   | `Ctrl+Z` / `Ctrl+Y` (`Ctrl+Shift+Z`) |
| Duplicar la selección                                | `Ctrl+D`                             |
| Eliminar la selección                                | `Delete`                             |
| Mover ligeramente la selección                       | `Teclas de flecha` (`Shift`: mayor)  |
| Espejar la selección verticalmente / horizontalmente | `M+V` / `M+H`                        |
| Alternar el modo de construcción                     | `G+N`                                |
| Cancelar la operación o deseleccionar                | `Escape`                             |
| Ajustar la vista al contenido                        | `1`                                  |

El espejado opera en el sitio a través del centro de la caja delimitadora
de la selección; las restricciones que cruzan el límite de la selección se
eliminan y las restricciones internas se conservan. Las copias obtienen
identificadores nuevos y restricciones internas reasignadas; deshacer las
elimina.

Además, cada herramienta de dibujo y modificación tiene un atajo de dos
teclas, documentado en su página:

| Herramienta                                                          | Atajo       |
| -------------------------------------------------------------------- | ----------- |
| [Trazado](path.md)                                                   | `G+P`       |
| [Arco](arc-ellipse.md)                                               | `G+A`       |
| [Elipse](arc-ellipse.md)                                             | `G+C`       |
| [Rectángulo](rectangle.md)                                           | `G+R`       |
| [Rectángulo redondeado](rectangle.md)                                | `G+O`       |
| [Rellenar área](fill.md)                                             | `G+F`       |
| [Cuadro de texto](expressions.md#template-expressions-in-text-boxes) | `G+T`       |
| [Arreglo Circular](arrays.md)                                        | `G+Y`       |
| [Arreglo a lo largo de curva](arrays.md)                             | `G+W`       |
| [Cuadrícula](grid.md)                                                | `G+G`       |
| [Desplazamiento](offset.md)                                          | `O+F`       |
| [Chaflán](chamfer-fillet.md)                                         | `C+H`       |
| [Redondeo](chamfer-fillet.md)                                        | `C+F`       |

Los atajos de restricciones están enumerados en la página
[Restricciones](constraints.md).

## Cuadrícula y ajuste {#grid-and-snapping}

El lienzo muestra una cuadrícula adaptativa cuyo espaciado se ajusta al nivel
de zoom y que está etiquetada a lo largo de los ejes en sus unidades
preferidas, por lo que también sirve de regla: puede leer tamaños y posiciones
directamente del lienzo.

Mientras dibuja o arrastra, el *ajuste magnético* atrae el cursor hacia los
puntos de referencia cercanos. El lienzo indica a qué se atrae el cursor:

- un **círculo azul** marca un punto existente (extremo),
- las **flechas verdes** marcan un punto medio,
- un **resaltado rosa** significa que el cursor está sobre un borde,
- las **líneas discontinuas** a lo largo del lienzo son guías de alineación,
  que aparecen cuando el cursor se alinea horizontal o verticalmente con otro
  punto,
- otros indicadores cubren casos especiales como espaciados equidistantes
  (naranja), tangencia (violeta) y centros (rojo).

El ajuste no es solo una ayuda visual: asentar la geometría sobre un objetivo
de ajuste crea la restricción correspondiente automáticamente. Terminar una
línea en un extremo existente hace que ambos puntos sean coincidentes; ajustar
a un punto medio crea una restricción de simetría; las guías de alineación se
convierten en restricciones horizontales o verticales. Si prefiere colocación
libre, `Tab` desactiva el ajuste magnético. Mantener pulsado `Shift` mientras
arrastra restringe el movimiento al eje más cercano.

![Guías de alineación y el indicador de ajuste equidistante al dibujar](/screenshots/addons-sketcher-snap.webp)

## Geometría de construcción {#construction-geometry}

Cualquier entidad puede marcarse como geometría de construcción. Las entidades
de construcción se dibujan discontinuas, actúan como guías de composición para
el resolvedor igual que cualquier otra geometría, y se excluyen de las
trayectorias de herramienta cuando se fabrica el boceto. Son útiles para
líneas de centro, círculos de construcción y el armazón detrás de diseños
simétricos. Seleccione una o más entidades y pulse `G+N` (o use la entrada
Construcción del menú circular) para alternar la marca; el alternador de
construcción de la barra de herramientas los oculta cuando estorban.

## Dónde ir a continuación

Las herramientas de dibujo están documentadas cada una en su propia página:
[Trazado](path.md) (líneas y curvas Bézier), [Arco y
elipse](arc-ellipse.md), [Rectángulo](rectangle.md) (y rectángulos
redondeados), [Rellenar áreas](fill.md) y [Cuadrícula](grid.md). Las
modificaciones como [Desplazamiento](offset.md) y [Chaflán y
redondeo](chamfer-fillet.md) dan nueva forma a la geometría existente,
[Arreglos](arrays.md) la copia a lo largo de un círculo o una curva, y
[Expresiones](expressions.md) explica los parámetros, las expresiones y los
cuadros de texto paramétricos. Los bocetos se pueden guardar y volver a
importar con todas las restricciones intactas; consulte
[Importación y exportación](import-export.md).
