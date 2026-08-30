---
description: "Dibuje rectángulos y rectángulos redondeados en el diseñador de Rayforge, con puntos centrales, teclas modificadoras e introducción de dimensiones."
---

# Rectángulo y rectángulo redondeado

El diseñador ofrece dos herramientas de rectángulo que comparten los
mismos gestos y teclas modificadoras: la herramienta **rectángulo** (`G+R`)
y la herramienta **rectángulo redondeado** (`G+O`).

![Un rectángulo y un rectángulo redondeado](/screenshots/addons-sketcher-tool-rectangle.webp)

## Dibujar rectángulos

Dibuje un rectángulo indicando dos esquinas opuestas, o pulse en la
primera esquina, arrastre y suelte en la esquina opuesta. Las teclas
modificadoras funcionan igual para ambas herramientas:

- Mantenga `Shift` para colocar el rectángulo de forma simétrica alrededor
  del punto de inicio.
- Mantenga `Ctrl` para restringirlo a un cuadrado.

Cada rectángulo crea automáticamente un **punto central** restringido al
centro geométrico, de modo que puede acotar o ajustar al punto medio de la
forma.

Mientras una vista previa está activa, puede teclear el tamaño exacto: la
barra de estado muestra los campos `W` y `H` (además de `R` para el radio
de esquina de los rectángulos redondeados). Teclee un valor, pulse `Tab`
para moverse entre los campos, y `Enter` para aplicarlo. Ambas
herramientas aceptan indistintamente el gesto de dos clics y el de clic y
arrastre; `Esc` cancela la vista previa.

El radio de esquina del rectángulo redondeado también puede cambiarse más
tarde editando sus restricciones — las esquinas están completamente
restringidas, de modo que el radio permanece ajustable.
