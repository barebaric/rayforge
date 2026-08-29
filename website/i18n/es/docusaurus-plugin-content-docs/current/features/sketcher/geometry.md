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
- **Elipses**: Cree elipses (y círculos) con dos clics: el primero define el
  punto central y el segundo el punto del borde. También puede pulsar en el
  centro, arrastrar y soltar en el borde; ambos gestos funcionan de manera
  intercambiable. Mantenga pulsado `Ctrl` para restringir a un círculo
  perfecto y `Shift` para usar el punto inicial como centro de la elipse.
- **Rectángulos**: Dibuje rectángulos especificando dos esquinas opuestas, o
  pulse en la primera esquina, arrastre y suelte en la esquina opuesta.
  Cada rectángulo crea automáticamente un punto central (restringido al
  centro geométrico) para que pueda dimensionar o ajustar a él. Mantenga
  pulsado `Shift` mientras dibuja para colocar el rectángulo simétricamente
  alrededor del punto inicial, y `Ctrl` para restringirlo a un cuadrado.
- **Rectángulos redondeados**: Dibuje rectángulos con esquinas redondeadas
  usando los mismos gestos y modificadores que la herramienta de rectángulo:
  dos clics o clic-y-arrastre, con `Shift` para centrar en el punto inicial
  y `Ctrl` para restringir a un cuadrado. El radio de las esquinas se puede
  ajustar escribiendo dimensiones (`0-9`, campos W, H y R).
- **Cuadros de texto**: Añada elementos de texto a su boceto. El contenido del
  texto soporta expresiones de plantilla paramétricas (vea
  [Plantillas de texto](../text.md)).
- **Rellenos**: Rellene regiones cerradas para crear áreas sólidas

Estos elementos forman la base de sus diseños 2D y pueden combinarse para crear
formas complejas. Los rellenos son especialmente útiles para crear regiones
sólidas que se grabarán o cortarán como una sola pieza.

## Dos clics o arrastre

Las herramientas de creación de formas (elipse, rectángulo, rectángulo
redondeado) aceptan dos gestos de manera intercambiable: haga clic en el
primer punto, mueva y haga clic en el segundo, o pulse en el primer punto,
arrastre y suelte en el segundo. Un clic rápido sin movimiento simplemente
activa la herramienta y espera el segundo punto, por lo que los clics
accidentales nunca dejan geometría degenerada. Mientras haya una vista
previa activa, la barra de estado muestra las teclas modificadoras
disponibles, y `Esc` cancela la vista previa.

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
