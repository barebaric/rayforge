---
description: "Aprenda sobre las restricciones geométricas y dimensionales en el diseñador paramétrico 2D de Rayforge."
---

# Sistema de restricciones paramétricas

El sistema de restricciones es el núcleo del diseñador paramétrico, permitiéndole
definir relaciones geométricas precisas:

## Restricciones geométricas

- **Coincidencia**: Fuerza dos puntos a ocupar la misma posición
- **Vertical**: Restringe una línea para que sea perfectamente vertical
- **Horizontal**: Restringe una línea para que sea perfectamente horizontal
- **Tangente**: Hace que una línea sea tangente a un círculo o arco
- **Perpendicular**: Fuerza dos líneas, una línea y un arco/círculo, o dos
  arcos/círculos a encontrarse en un ángulo de 90 grados
- **Punto sobre línea/forma**: Restringe un punto para que se encuentre sobre
  una línea, arco o círculo
- **Colineal**: Fuerza dos o más líneas a encontrarse sobre la misma línea
  infinita
- **Simetría**: Crea relaciones simétricas entre elementos. Admite dos modos:
  - **Simetría de punto**: Seleccione 3 puntos (el primero es el centro)
  - **Simetría de línea**: Seleccione 2 puntos y 1 línea (la línea es el eje)

## Restricciones dimensionales

- **Distancia**: Establece la distancia exacta entre dos puntos o a lo largo de
  una línea
- **Diámetro**: Define el diámetro de un círculo
- **Radio**: Establece el radio de un círculo o arco
- **Ángulo**: Exige un ángulo específico entre dos líneas
- **Relación de aspecto**: Fuerza la proporción entre dos distancias a ser igual
  a un valor especificado
- **Igual longitud/radio**: Fuerza múltiples elementos (líneas, arcos, elipses
  o círculos) a tener la misma longitud o radio
- **Igual distancia**: Hace que dos segmentos de línea tengan la misma longitud
  (diferente de Igual longitud/radio, que también puede aplicarse a arcos y
  círculos)
