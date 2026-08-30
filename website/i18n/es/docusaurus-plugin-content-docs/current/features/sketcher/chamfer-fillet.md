---
description:
  "Biselar esquinas vivas con la herramienta de chaflán o redondearlas con la herramienta de
  redondeo en el diseñador de Rayforge."
---

# Chaflán y redondeo

El diseñador proporciona dos herramientas para modificar esquinas donde se encuentran dos líneas:

- **Chaflán** (`C+H`): reemplaza una esquina aguda por un borde biselado.
- **Redondeo** (`C+F`): reemplaza una esquina aguda por un borde redondeado.

![Un rectángulo achaflanado junto a un rectángulo con redondeo](/screenshots/addons-sketcher-tool-chamfer-fillet.webp)

Para aplicar una de ellas:

1. Seleccione un punto de unión donde se encuentren exactamente dos líneas.
2. Pulse `C+H` para chaflán o `C+F` para redondeo, o elija la herramienta en el menú circular.

La esquina se reemplaza en un solo paso. Las dos líneas se recortan y el borde nuevo se inserta
entre ellas, junto con restricciones que mantienen los segmentos recortados colineales con los
originales y la esquina simétrica. En un chaflán, la longitud del bisel toma por defecto una
fracción de la línea adyacente más corta; en un redondeo, el radio del arco se elige para encajar.
Arrastrar después los extremos del borde insertado ajusta su tamaño, con las restricciones
manteniendo la esquina intacta.
