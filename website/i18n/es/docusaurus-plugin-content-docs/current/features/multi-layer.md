---
description: "Organiza trabajos láser en capas con diferentes ajustes. Gestiona el orden de corte, operaciones y materiales con el sistema multicapa de Rayforge."
---

# Flujo de trabajo multicapa

![Panel de capas](/screenshots/bottom-panel-layers.png)

El sistema multicapa de Rayforge te permite organizar trabajos en etapas
de procesamiento separadas. Cada capa es un contenedor para piezas de
trabajo y tiene su propio flujo de trabajo — una secuencia de pasos,
cada uno con ajustes de láser independientes.

:::tip Cuándo no necesitas múltiples capas
En muchos casos, una sola capa es suficiente. Cada paso dentro de una
capa tiene sus propios ajustes de láser, potencia, velocidad y otros
parámetros, por lo que puedes grabar y contornear en la misma capa.
Solo necesitas capas separadas cuando quieres contornear diferentes
partes de una imagen con distintos ajustes, o cuando necesitas
diferentes configuraciones de WCS o rotatorio.
:::

## Crear y gestionar capas

### Añadir una capa

Haz clic en el botón **+** del panel de capas. Los documentos nuevos
comienzan con tres capas vacías.

### Reordenar capas

Arrastra y suelta capas en el panel para cambiar el orden de ejecución.
Las capas se procesan de izquierda a derecha. Puedes usar **arrastrar
con clic central** para desplazarte dentro de la lista de capas.

### Reordenar piezas de trabajo

Las piezas de trabajo dentro de una capa se pueden reorganizar
arrastrando y soltando para controlar su orden Z.

### Eliminar una capa

Selecciona la capa y haz clic en el botón de eliminar. Todas las piezas
de trabajo de la capa se eliminan. Puedes deshacer la eliminación si lo
necesitas.

## Propiedades de capa

Cada capa tiene los siguientes ajustes, disponibles a través del icono
de engranaje en la columna de la capa:

- **Nombre** — se muestra en la cabecera de la capa
- **Color** — se usa para renderizar las operaciones de la capa en el
  lienzo
- **Visibilidad** — el icono del ojo alterna si la capa se muestra en
  el lienzo y en las vistas previas. Las capas ocultas siguen incluidas
  en el G-code generado.
- **Sistema de coordenadas (WCS)** — asigna un sistema de coordenadas
  de trabajo a esta capa. Cuando se establece en un WCS específico
  (p. ej. G54, G55), la máquina cambia a ese sistema de coordenadas al
  inicio de la capa. Selecciona **Predeterminado** para usar el WCS
  global.
- **Modo rotatorio** — activa el modo de accesorio rotatorio para esta
  capa, permitiéndote mezclar trabajo de cama plana y cilíndrico en el
  mismo proyecto. Configura el módulo rotatorio y el diámetro del
  objeto en los ajustes de la capa.

## Flujos de trabajo por capa

Cada capa tiene un **flujo de trabajo** — una secuencia de pasos que se
muestra como una tubería de iconos en la columna de la capa. Cada paso
define una única operación (p. ej. contorno, grabado ráster) con sus
propios ajustes de láser, potencia, velocidad y otros parámetros.

Haz clic en un paso para configurarlo. Usa el botón **+** de la tubería
para añadir más pasos a una capa. Los pasos se pueden reordenar
arrastrando y soltando.

## Importación de archivos vectoriales

Al importar archivos vectoriales (SVG, DXF, PDF), el diálogo de
importación ofrece tres formas de manejar las capas del archivo de
origen:

- **Asignar a capas existentes** — importa cada capa de origen a la
  capa del documento correspondiente por posición
- **Nuevas capas** — crea una nueva capa del documento por cada capa de
  origen
- **Aplanar** — importa todo a la capa activa

Al usar **Asignar a capas existentes** o **Nuevas capas**, el diálogo
muestra una lista de las capas del archivo de origen con interruptores
para seleccionar cuáles importar.

## Asignar piezas de trabajo a capas

**Arrastrar y soltar:** Selecciona pieza(s) de trabajo en el lienzo o
panel de documento y arrástralas a la capa destino.

**Cortar y pegar:** Corta una pieza de trabajo de la capa actual
(Ctrl+X), selecciona la capa destino y pega (Ctrl+V).

## Orden de ejecución

Durante un trabajo, las capas se procesan de izquierda a derecha.
Dentro de cada capa, se procesan todas las piezas de trabajo antes de
pasar a la siguiente capa. El flujo de trabajo estándar es grabar
primero y cortar al final, para que las piezas permanezcan en su sitio
durante el grabado.

## Páginas relacionadas

- [Operaciones](./operations/contour) - Tipos de operaciones para
  flujos de trabajo por capa
- [Modo de simulación](./simulation-mode) - Vista previa de ejecución
  multicapa
- [Macros y Hooks](../machine/hooks-macros) - Hooks a nivel de capa
  para automatización
- [Vista previa 3D](../ui/3d-preview) - Visualizar la pila de capas
