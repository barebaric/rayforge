# Limpieza de rutas

Los logotipos y dibujos de otros programas no siempre están limpios. Un contorno puede terminar una
fracción de milímetro antes de su punto inicial, una forma puede estar hecha de varias piezas
abiertas o la misma ruta puede estar dos veces en el archivo. Rayforge corta esas rutas tal como
son: una forma que solo parece cerrada se trata como abierta y una ruta duplicada se corta dos
veces.

Las herramientas de limpieza de rutas corrigen esto en las piezas de trabajo seleccionadas. Las
encontrarás en el menú **Objeto** y en **Limpiar rutas** del menú contextual del lienzo. Cada
herramienta es un solo paso de deshacer.

## Cerrar rutas

**Cerrar rutas…** cierra cada ruta abierta cuyo punto inicial y final están más cerca que la
tolerancia que indiques. El hueco se salva con una línea recta entre los dos puntos existentes, así
que la forma no se mueve ni cambia de tamaño.

## Unir rutas abiertas

**Unir rutas abiertas…** une en rutas más largas las rutas abiertas cuyos extremos están más cerca
que la tolerancia. Las piezas se invierten cuando hace falta, así que el orden y la dirección en que
se dibujaron no importan. Si después los extremos de una ruta unida coinciden, la ruta también se
cierra.

Ambas herramientas recuerdan la última tolerancia que usaste durante la sesión. Una tolerancia entre
0,05 mm y 0,2 mm funciona para la mayoría de los archivos importados.

## Eliminar duplicados

**Eliminar duplicados** quita las rutas que están exactamente encima de otra ruta de la misma pieza
de trabajo, sin importar su dirección o punto inicial. Si seleccionas varias piezas de trabajo,
también se quita una pieza que sea una copia exacta de otra pieza seleccionada en la misma posición.

Las rutas que solo se superponen en parte de su longitud se mantienen. De ellas se encarga el
posprocesador [Fusionar líneas](merge-lines) al generar el trabajo.

## Separar

**Separar** convierte cada ruta de una pieza de trabajo en una pieza propia, incluidos los agujeros
y las rutas abiertas. Esto es distinto de **Dividir**, que mantiene una isla junto con sus agujeros.

La limpieza de rutas solo funciona en piezas de trabajo importadas y trazadas. Las piezas que vienen
de un boceto se editan en el [Sketcher](sketcher/index).

## Páginas relacionadas

- [Fusionar líneas](merge-lines) - Cortar los segmentos superpuestos una sola vez
- [Herramientas del lienzo](../ui/canvas-tools) - Seleccionar y eliminar segmentos individuales
- [Importar archivos](../files/importing) - Llevar diseños a Rayforge
