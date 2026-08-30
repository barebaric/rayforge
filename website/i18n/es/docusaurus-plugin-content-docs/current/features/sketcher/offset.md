---
description:
  "Amplíe, reduzca o convierta contornos en ranuras con la herramienta de desplazamiento en el
  diseñador de Rayforge."
---

# Desplazamiento de contorno

La herramienta de desplazamiento (`O+F`) amplía o reduce un contorno seleccionado una distancia
dada, o expande una ruta abierta en una ranura. Seleccione las entidades que forman un contorno (o
use doble clic para seleccionar la geometría conectada), y después pulse `O+F`, o use la entrada
**Desplazamiento** del menú circular.

![Diálogo de desplazamiento de contorno](/screenshots/addons-sketcher-offset-dialog.webp)

El diálogo pide la distancia de desplazamiento y muestra una vista previa en vivo del resultado en
el lienzo mientras teclea:

- Los **contornos cerrados** se amplían con una distancia positiva y se reducen con una negativa. Se
  rechaza desplazar más allá del punto donde el contorno colapsaría.
- Las **rutas abiertas** se convierten en un contorno cerrado en forma de ranura del ancho indicado,
  con extremos redondeados.

![Contorno Bézier](/screenshots/addons-sketcher-offset-before.webp)
![Contorno Bézier desplazado convertido en ranura](/screenshots/addons-sketcher-offset-after.webp)

Desplazar reemplaza el contorno seleccionado por el resultado:

- Los círculos, arcos y elipses aislados conservan su tipo de entidad y se actualizan en el sitio,
  de modo que siguen siendo editables y restringibles como antes.
- Las cadenas de segmentos conectados (incluidas las Bézier) se reemplazan por una entidad polígono.
  El polígono se edita como un todo: arrastre su punto central para moverlo, y el punto de tirador
  para rotarlo o escalarlo uniformemente.

Si la selección contiene varios contornos desconectados, cada uno se desplaza de forma independiente
en un solo paso.
