---
description:
  "Coloque texto grabado, etiquetas y números de serie en un boceto con la herramienta de cuadro de
  texto de Rayforge."
---

# Cuadro de texto

La herramienta de cuadro de texto (`G+T`) coloca texto en el boceto como geometría editable — texto
grabado, etiquetas y números de serie. Los cuadros de texto son totalmente paramétricos: los glifos
viven dentro de un marco con restricciones, de modo que se vuelven a resolver cada vez que el marco
se mueve o se dimensiona.

![Una marca denominativa y una etiqueta de pieza](/screenshots/addons-sketcher-tool-text-box.webp)

## Crear y editar texto

1. Seleccione la herramienta de cuadro de texto en el menú circular, en el menú **Boceto**, o con
   `G+T`.
2. Haga clic donde desea que comience el texto: aparece un cuadro de texto en el punto de clic y la
   herramienta pasa directamente al modo de edición.
3. Escriba el texto — el cuadro se redimensiona solo para adaptarse mientras escribe.
4. Pulse `Enter` o `Escape` para terminar la edición.

Para editar un cuadro de texto existente, haga clic dentro de él. Un doble clic selecciona una
palabra, un triple clic la línea completa, y el texto puede seleccionarse y reemplazarse como en
cualquier editor de texto, incluidos `Ctrl+C`/`Ctrl+V`, deshacer/rehacer y pegar a mitad de la
edición.

## Propiedades de fuente

![El panel de propiedades de fuente](/screenshots/addons-sketcher-tool-text-box-font-properties.webp)

El panel **Propiedades de fuente** de la barra lateral controla el aspecto del cuadro de texto
seleccionado en el lienzo:

- **Familia de fuente** — elija entre las fuentes del sistema instaladas.
- **Tamaño de fuente** — en puntos.
- Alternadores **Negrita** y **Cursiva**.

## Un marco paramétrico

Un cuadro de texto no es una imagen rasterizada: sus glifos son geometría de boceto real, dispuesta
dentro de un marco definido por un origen y puntos de anchura y altura. El marco se dibuja
discontinuo como geometría de construcción, de modo que sirve de referencia de composición y nunca
acaba en las trayectorias de herramienta cuando se fabrica el boceto. Como todo lo demás en el
diseñador, el marco tiene restricciones, por lo que puede dimensionarse como cualquier otra
geometría — cambie la restricción de anchura y el texto se vuelve a resolver para llenar el cuadro.

Hacer clic dentro de un cuadro de texto con la [herramienta de relleno](fill.md) alterna el relleno
de los glifos del texto en lugar de crear un relleno de región.

## Expresiones de plantilla

Los cuadros de texto aceptan **expresiones de plantilla**: todo lo que esté entre llaves se evalúa
cuando el boceto se resuelve, de modo que las etiquetas pueden mostrar valores en vivo como
dimensiones, fechas o números de serie únicos. Consulte
[Expresiones de plantilla en cuadros de texto](expressions.md#template-expressions-in-text-boxes)
para conocer los detalles y las funciones integradas.
