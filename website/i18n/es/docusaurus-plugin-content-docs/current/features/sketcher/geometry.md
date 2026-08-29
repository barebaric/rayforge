---
description: "Aprenda a crear líneas, curvas Bézier, arcos, elipses, rectángulos y otra geometría 2D en el diseñador Rayforge."
---

# Creación de geometría 2D

El diseñador permite crear los siguientes elementos geométricos básicos:

- **Trazados (líneas y curvas Bézier)**: Dibuje líneas rectas y curvas Bézier
  suaves con la herramienta de trazado unificada. Haga clic para colocar puntos,
  arrastre para crear tiradores Bézier.
- **Arcos**: Dibuje arcos especificando un punto central, un punto de inicio y
  un punto final
- **Elipses**: Cree elipses (y círculos) definiendo un punto central y
  arrastrando para ajustar el tamaño y la proporción. Mantenga pulsado `Ctrl`
  mientras arrastra para restringir a un círculo perfecto.
- **Rectángulos**: Dibuje rectángulos especificando dos esquinas opuestas.
  Cada rectángulo crea automáticamente un punto central (restringido al
  centro geométrico) para que pueda dimensionar o ajustar a él. Mantenga
  pulsado `Shift` mientras dibuja para colocar el rectángulo simétricamente
  alrededor del punto inicial, igual que la herramienta de elipse.
- **Rectángulos redondeados**: Dibuje rectángulos con esquinas redondeadas
- **Cuadros de texto**: Añada elementos de texto a su boceto. El contenido del
  texto soporta expresiones de plantilla paramétricas (vea
  [Plantillas de texto](../text.md)).
- **Rellenos**: Rellene regiones cerradas para crear áreas sólidas

Estos elementos forman la base de sus diseños 2D y pueden combinarse para crear
formas complejas. Los rellenos son especialmente útiles para crear regiones
sólidas que se grabarán o cortarán como una sola pieza.

## Trabajar con curvas Bézier

La herramienta de trazado admite curvas Bézier para crear formas suaves y
orgánicas:

### Dibujar curvas Bézier

1. Seleccione la herramienta de trazado en el menú circular o use el atajo de
   teclado
2. Haga clic para colocar puntos; cada clic crea un nuevo punto
3. Arrastre tras hacer clic para crear tiradores Bézier y obtener curvas suaves
4. Siga añadiendo puntos para construir su trazado
5. Pulse Escape o haga doble clic para finalizar el trazado

### Editar curvas Bézier

- **Mover puntos**: Haga clic y arrastre cualquier punto para reposicionarlo
- **Ajustar tiradores**: Arrastre los extremos de los tiradores para modificar
  la forma de la curva
- **Conectar a puntos existentes**: Al editar un trazado, puede ajustarse a los
  puntos existentes de su boceto
- **Suavizar/simetrizar**: Los puntos conectados por una restricción de
  coincidencia pueden suavizarse (tangente continua) o simetrizarse (tiradores
  reflejados)

### Convertir curvas en líneas

Use la **herramienta de enderezamiento** para convertir curvas Bézier en líneas
rectas. Esto es útil cuando necesita geometría limpia y sencilla. Seleccione los
segmentos Bézier que desea convertir y aplique la acción de enderezamiento.
