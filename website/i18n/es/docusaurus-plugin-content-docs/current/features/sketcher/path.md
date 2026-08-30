---
description:
  "Dibuje líneas rectas y curvas Bézier suaves con la herramienta de trazado en el diseñador de
  Rayforge."
---

# Herramienta de trazado

La herramienta de trazado (`G+P` o `G+L`) dibuja cadenas conectadas de líneas rectas y curvas Bézier
suaves en un flujo de trabajo unificado. Es la herramienta de dibujo más versátil del diseñador:
haga clic para colocar puntos y arrastre para curvar el segmento.

![Un trazado de dos líneas unidas por un segmento Bézier, con sus puntos de paso y tiradores](/screenshots/addons-sketcher-tool-path.webp)

## Dibujar trazados

1. Seleccione la herramienta de trazado desde el menú circular, el menú **Boceto**, o con `G+P`.
2. Haga clic para colocar el primer punto. Una vista previa en vivo sigue al cursor.
3. Haga clic de nuevo sin arrastrar para terminar un segmento recto — el segmento siguiente comienza
   inmediatamente en ese punto.
4. Pulse en un punto y arrastre antes de soltar para convertir el segmento en una curva Bézier. El
   arrastre controla cuánto se arquea la curva.
5. Siga añadiendo puntos para construir su trazado.
6. Pulse `Escape` o haga doble clic para terminar el trazado.

Mientras una vista previa está activa, la barra de estado enumera las teclas modificadoras que se
aplican, y `Esc` la cancela.

## Trabajar con curvas Bézier

Las curvas Bézier crean formas suaves y orgánicas:

- **Ajustar tiradores**: seleccione una Bézier y arrastre los extremos de los tiradores redondos
  para modificar la forma de la curva. Cada tirador curva la Bézier en su lado del punto de paso.
- **Conectar con puntos existentes**: mientras dibuja, el ajuste magnético une los segmentos nuevos
  a los puntos existentes de su boceto, y la restricción correspondiente se crea automáticamente.

### Tipos de punto de paso

El punto donde se encuentran dos segmentos de un trazado es un _punto de paso_. El tipo de punto de
paso controla cómo fluye la curva a través de él:

- **Agudo**: los tiradores a ambos lados son independientes, lo que produce una esquina.
- **Suave**: los tiradores comparten una tangente, lo que produce una transición continua y
  redondeada.
- **Simétrico**: como Suave, pero los tiradores además se reflejan, de modo que ambos lados se
  curvan por igual.

Para cambiar el tipo de un punto de paso, haga clic derecho sobre él (o sobre el segmento Bézier
contiguo) y elija el tipo en el menú circular. Los puntos de paso Bézier recién dibujados son
simétricos.

![El menú circular sobre un punto de paso Bézier seleccionado, con las herramientas Enderezar, Agudo, Suave y Simétrico](/screenshots/addons-sketcher-tool-path-pie-menu.webp)

### Convertir curvas en líneas

La herramienta **Enderezar** del mismo menú circular convierte las curvas Bézier de nuevo en líneas
rectas, algo útil cuando necesita geometría limpia y simple. Seleccione los segmentos Bézier que
desea convertir y aplique la acción de enderezar. Los segmentos se reducen a la conexión recta entre
sus extremos.

## Restricciones automáticas

La herramienta de trazado participa en el ajuste magnético como todas las demás herramientas de
dibujo. Cuando las guías de ajuste muestran alineación durante el dibujo, las restricciones
horizontales y verticales correspondientes se crean automáticamente, lo que mantiene el boceto
ordenado desde el principio en lugar de arreglarlo después. Mantenga `Shift` para restringir el
segmento nuevo al eje más cercano. Vea [Cuadrícula y ajuste](index.md#grid-and-snapping) para la
lista completa de indicadores de ajuste.
