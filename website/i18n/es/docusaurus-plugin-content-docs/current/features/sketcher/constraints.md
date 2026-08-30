---
description: "Cómo funcionan las restricciones en el diseñador de Rayforge: añadirlas, editarlas, seleccionarlas y eliminarlas, y resolver conflictos."
---

# Restricciones

Las restricciones son las reglas que mantienen unido un boceto. Cada una es una
pequeña afirmación sobre la geometría — «estos dos puntos son uno y el mismo»,
«esta línea mide exactamente 80 mm» — y después de cada edición el resolvedor
reordena el boceto para que todas las afirmaciones se cumplan a la vez. La
geometría sin restricciones puede desplazarse libremente; cada restricción que
añade fija un grado de libertad.

Hay dos familias. Las **restricciones geométricas** capturan relaciones que no
llevan ninguna medida: coincidencia, horizontalidad, tangencia, simetría. Las
**restricciones dimensionales** asocian un número a la geometría: una
distancia, un radio, un ángulo. Los valores dimensionales aceptan expresiones
(vea [más abajo](#editar-valores-dimensionales)), y ahí es donde ocurre lo
"paramétrico" del dibujo paramétrico.

El resolvedor informa de su estado mediante colores. La geometría sujeta por
restricciones se dibuja en verde, los puntos sin restringir en negro, y un
boceto completamente restringido oscurece el verde. Los marcadores de
restricciones válidas son verdes, los de expresiones naranjas, y los
marcadores de restricciones que el resolvedor no puede satisfacer se ponen
rojos (vea [conflictos](#cuando-las-restricciones-entran-en-conflicto)).

## Añadir una restricción

Seleccione la geometría a la que debe aplicarse la restricción y, a
continuación, pulse el atajo de teclado o elija la restricción en el menú
circular: las restricciones geométricas están en el grupo **Restringir** y las
dimensionales en el grupo **Acotar**. Cada restricción exige una selección
concreta:

| Restricción                     | Selección                       | Atajo      |
| ------------------------------- | ------------------------------- | ---------- |
| Horizontal / Vertical           | 2 puntos, o cualquier línea     | `H` / `V`  |
| Coincidente / Punto sobre forma | 2 puntos, o un punto + una forma| `O` o `C`  |
| Perpendicular                   | 2 formas                        | `N`        |
| Tangente                        | 1 línea + 1 arco o círculo      | `T`        |
| Simetría                        | 3 puntos, o 2 puntos + 1 línea  | `S`        |
| Igual longitud                  | 2 o más formas                  | `E`        |
| Distancia                       | 2 puntos, o 1 línea             | `K+D`      |
| Diámetro                        | 1 círculo                       | `K+O`      |
| Radio                           | 1 arco o círculo                | `K+R`      |
| Ángulo                          | 2 líneas                        | `K+A`      |
| Relación de aspecto             | 2 líneas                        | `K+X`      |

El orden de una selección nunca importa, con una excepción: con tres puntos
seleccionados, la Simetría usa el **último** punto como centro de espejo. Un
atajo solo se activa cuando la selección actual encaja con la restricción;
todo lo demás se filtra también del menú circular.

Las restricciones también aparecen por sí solas mientras dibuja: ajustar a un
extremo crea una restricción coincidente, y las guías de alineación se
convierten en restricciones horizontales o verticales (vea
[la descripción general del diseñador](index.md#grid-and-snapping)).

## Restricciones geométricas

Una restricción **coincidente** fusiona dos puntos distintos en una sola
ubicación. Seleccione los dos puntos y ambos se juntan; el marcador es un anillo
alrededor del punto unido. Dibujar una línea que termina exactamente en un
extremo existente crea esta restricción automáticamente.

![Dos líneas unidas por una restricción coincidente](/screenshots/addons-sketcher-constraint-coincident.webp)

**Horizontal** y **Vertical** giran la línea seleccionada, o el par de puntos
seleccionados, sobre un eje. Los marcadores son pequeñas barras, horizontal y
vertical respectivamente, dibujadas junto a la geometría.

![Una restricción horizontal](/screenshots/addons-sketcher-constraint-horizontal.webp)

![Una restricción vertical](/screenshots/addons-sketcher-constraint-vertical.webp)

**Perpendicular** fuerza dos formas a encontrarse en ángulo recto. Funciona
para dos líneas, una línea y un arco o círculo, o dos arcos y círculos. El
marcador es un arco en ángulo recto en la intersección.

![Dos líneas que se encuentran en ángulo recto](/screenshots/addons-sketcher-constraint-perpendicular.webp)

**Tangente** suaviza la transición donde una línea encuentra un arco o
círculo: la línea se gira para tocar la curva sin cruzarla. Su marcador es una
pequeña "T" en el punto de contacto.

![Una línea tangente a un círculo](/screenshots/addons-sketcher-constraint-tangent.webp)

**Punto sobre forma** fija un punto sobre una línea, arco o círculo, sin
fusionarlo con ningún punto concreto como hace la coincidente. Seleccione un
punto y una forma; el marcador es un anillo alrededor del punto restringido.
Cuando la forma es una curva (Bézier), el punto queda restringido a
deslizarse a lo largo de ella.

![El extremo de una línea apoyado en otra línea](/screenshots/addons-sketcher-constraint-point-on-line.webp)

**Simetría** refleja dos puntos respecto a un centro o un eje, y presenta los
dos modos ya mencionados: seleccione tres puntos y el último se convierte en
el centro alrededor del cual se reflejan los dos primeros, o seleccione dos
puntos y una línea para reflejarlos a través de esa línea. El marcador es un
par de puntas de flecha opuestas en el punto medio entre los puntos
reflejados.

![Dos puntos reflejados respecto a una línea](/screenshots/addons-sketcher-constraint-symmetry.webp)

Una séptima restricción geométrica, la **colineal**, fuerza a los puntos a
estar sobre una misma línea infinita. No tiene marcador en el lienzo y no
puede aplicarse a mano: las herramientas de chaflán y redondeo la crean para
mantener alineada la esquina modificada.

## Restricciones dimensionales

La restricción de **distancia** fija la separación entre dos puntos o la
longitud de una línea. Su etiqueta muestra el valor actual en el punto medio
del tramo medido; cuando los dos puntos no están ya unidos por una línea, una
línea guía discontinua deja claro qué se está midiendo.

![Una restricción de distancia de 80 mm](/screenshots/addons-sketcher-constraint-distance.webp)

Los círculos y arcos tienen sus propias cotas. **Diámetro** etiqueta el ancho
completo de un círculo con el prefijo `Ø`, **radio** etiqueta la distancia
desde el centro de un arco o círculo con el prefijo `R`, y ambos colocan la
etiqueta justo fuera de la forma con una guía corta.

![Una restricción de diámetro](/screenshots/addons-sketcher-constraint-diameter.webp)

![Una restricción de radio](/screenshots/addons-sketcher-constraint-radius.webp)

La restricción de **ángulo** establece el ángulo entre dos líneas
seleccionadas. Dibuja un arco entre las dos direcciones en su intersección,
etiquetado con el valor en grados.

![Una restricción de ángulo de 45 grados](/screenshots/addons-sketcher-constraint-angle.webp)

La **relación de aspecto** vincula las longitudes de dos líneas: la longitud
de la primera dividida por la longitud de la segunda debe ser igual al valor
indicado. Su marcador, un par de soportes de esquina opuestos, se sitúa en el
punto donde las líneas se encuentran.

![Una restricción de relación de aspecto entre dos líneas](/screenshots/addons-sketcher-constraint-aspect-ratio.webp)

Por último, la restricción de **igual longitud** aplicada a dos o más líneas,
arcos, círculos o elipses hace que todos compartan una misma longitud o radio,
marcando cada forma con un signo `=`. El resolvedor también usa internamente
una variante de igual distancia de esta restricción, por ejemplo para
mantener redondo un círculo o simétricos los dos lados de un chaflán, que
lleva el mismo marcador `=` pero no puede aplicarse a mano.

![Dos líneas de igual longitud](/screenshots/addons-sketcher-constraint-equal-length.webp)

## Editar valores dimensionales

Haga doble clic en la etiqueta de una restricción dimensional para editarla.
El diálogo acepta un número simple o una expresión: se puede hacer referencia
a los parámetros del boceto y a las variables de entrada por su nombre, y hay
funciones matemáticas disponibles: un radio de `width/2` sigue al parámetro de
anchura allá donde vaya. Una vez que una restricción está gobernada por una
expresión, su marcador se vuelve naranja como recordatorio de que el número
se calcula, no se teclea. La sintaxis completa, junto con los parámetros del
boceto a los que puede referirse, se describe en
[Expresiones](expressions.md).

Hacer doble clic en una línea, arco o círculo aún sin acotar ofrece crear la
cota correspondiente directamente (distancia, radio o diámetro).

## Seleccionar y eliminar

Los marcadores de restricciones participan en la selección como cualquier otra
cosa: al pasar el cursor se muestra un resaltado amarillo y una información
herramienta con el nombre de la restricción, y un clic la selecciona,
dibujándola en azul. Pulsar `Delete` elimina la restricción seleccionada y
libera la geometría que sujetaba. Al eliminar geometría, sus restricciones se
van con ella. Para las restricciones dimensionales, el diálogo de edición
descrito arriba no tiene botón de eliminación: quitar una cota es una
eliminación normal del marcador seleccionado.

## Cuando las restricciones entran en conflicto

Las restricciones que se contradicen entre sí — un triángulo cuyos lados no
pueden ser todos ciertos a la vez, por ejemplo — no pueden romper el boceto:
el resolvedor hace lo que puede y señala lo que no pudo satisfacer. Las
restricciones en conflicto se ponen rojas, tanto sus marcadores como la
geometría que sujetan, de modo que la zona dañada se ve de un vistazo.

![Restricciones de distancia en conflicto, señaladas en la barra lateral](/screenshots/addons-sketcher-conflicts.webp)

La barra lateral enumera cada conflicto en **Restricciones en conflicto**, y
cada fila nombra la restricción y los puntos que toca. Las filas son
interactivas: pasar el cursor sobre una resalta la restricción en el lienzo,
hacer clic en una la selecciona, y el botón de eliminar de la derecha la
quita. Normalmente, la forma más rápida de salir de un conflicto es eliminar
o recalcular la restricción que expresa la intención desfasada; la lista
existe precisamente porque el resolvedor no puede adivinar cuál de las reglas
contradictorias es la incorrecta.

## Dónde ir a continuación

Cada herramienta de dibujo está documentada en su propia página; vea
[Trazado](path.md), [Arco y elipse](arc-ellipse.md) y
[Rectángulo](rectangle.md) para saber cómo dibujar las formas a las que se
adjuntan estas restricciones.
