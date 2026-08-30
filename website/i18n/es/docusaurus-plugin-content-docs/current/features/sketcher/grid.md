---
description:
  "Cree una cuadrícula de construcción de filas y columnas como andamiaje de dibujo en el diseñador
  de Rayforge."
---

# Cuadrícula

La herramienta de cuadrícula (`G+G`) crea una cuadrícula homogénea de líneas de construcción — filas
y columnas de guías espaciadas uniformemente que sirven de andamiaje de dibujo, por ejemplo para
disponer un patrón de perforaciones o para alinear elementos repetidos.

![Una cuadrícula de construcción de 4x6](/screenshots/addons-sketcher-tool-grid.webp)

1. Seleccione la herramienta de cuadrícula en el menú circular, en el menú **Boceto**, o con `G+G`.
2. Un diálogo pide el número de **filas** y **columnas**.
3. Confirme para crear la cuadrícula en el origen del boceto con celdas de 10 mm.

La cuadrícula consiste en geometría de construcción: se dibuja discontinua, actúa como referencia de
ajuste y alineación como cualquier otra geometría, y se excluye de las trayectorias de herramienta
cuando se fabrica el boceto (vea [Geometría de construcción](index.md#construction-geometry)). Las
líneas individuales se pueden mover o eliminar como cualquier otra geometría, y seleccionarlas y
alternar el modo de construcción con `G+N` convierte el andamiaje en geometría real.
