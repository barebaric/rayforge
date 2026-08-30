---
description: "Dibuje arcos y elipses (incluidos círculos) en el diseñador de Rayforge, con teclas modificadoras e introducción de dimensiones."
---

# Arco y elipse

El diseñador proporciona dos herramientas de formas curvas: la
**herramienta de arco** para arcos circulares y la **herramienta de
elipse** para elipses y círculos.

![Un arco y una elipse tal como los crean sus herramientas](/screenshots/addons-sketcher-tool-arc-ellipse.webp)

## Herramienta de arco

La herramienta de arco (`G+A`) crea un arco en tres clics:

1. Haga clic en el punto **central**.
2. Haga clic en el punto de **inicio** — su distancia al centro fija el
   radio.
3. Mueva el cursor para previsualizar el arco barriendo entre los dos
   puntos y haga clic en la posición **final**.

Mientras la vista previa está activa, puede teclear un número para fijar
el radio con exactitud; pulse `Tab` o `Enter` para aplicarlo. `Tab` antes
de teclear alterna el ajuste magnético.

## Herramienta de elipse

La herramienta de elipse (`G+C`) crea elipses y círculos con dos clics: el
primero fija el centro, el segundo fija el punto del borde. También puede
pulsar en el centro, arrastrar y soltar en el borde — ambos gestos
funcionan indistintamente.

- Mantenga `Ctrl` para restringir la forma a un círculo perfecto.
- Mantenga `Shift` para usar el punto de inicio como el centro de la
  elipse.

## Dos clics o arrastrar

Como las herramientas de [rectángulo](rectangle.md), la herramienta de
elipse acepta dos gestos indistintamente: haga clic en el primer punto,
mueva y haga clic en el segundo, o pulse en el primer punto, arrastre y
suelte en el segundo. Un clic rápido sin movimiento simplemente arma la
herramienta y espera el segundo punto, de modo que los clics perdidos
nunca dejan geometría degenerada detrás. Mientras una vista previa está
activa, la barra de estado muestra las teclas modificadoras disponibles, y
`Esc` cancela la vista previa.
