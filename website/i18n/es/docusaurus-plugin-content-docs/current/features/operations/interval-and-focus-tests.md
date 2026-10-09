---
description:
  "Encuentra el mejor intervalo de línea para grabar y el punto focal de tu láser con la prueba de
  intervalo y la prueba de enfoque."
---

# Pruebas de intervalo y de enfoque

Junto a la [cuadrícula de prueba de material](material-test-grid.md), el menú Herramientas ofrece
dos trabajos de calibración más. Ambos se crean como una capa normal con piezas y operaciones: puedes
moverla, encuadrarla y previsualizarla como cualquier otro contenido, y cambiar después los ajustes
de cada celda o línea en su operación.

## Prueba de intervalo

**Herramientas → Crear prueba de intervalo** graba una fila de cuadrados rellenos. Cada cuadrado
tiene su propia operación **Grabar** con su propio intervalo de línea, repartido de forma uniforme
entre el intervalo menor y el mayor que indiques, mientras la potencia y la velocidad son las mismas
para todos. Las etiquetas bajo cada cuadrado muestran el intervalo en milímetros y la densidad de
líneas correspondiente en líneas por pulgada (LPI).

Elige el cuadrado que esté relleno de forma uniforme, sin líneas visibles y sin quemarse demasiado,
y usa su intervalo para grabar en ese material. Las etiquetas se cortan antes que los cuadrados con
una operación propia de baja potencia.

## Prueba de enfoque

**Herramientas → Crear prueba de enfoque** encuentra la altura a la que el haz es más nítido. Un
desplazamiento positivo significa más distancia entre el cabezal y el material. La línea más fina
marca el mejor enfoque.

| Método                            | Cómo cambia la altura                                                                              |
| --------------------------------- | -------------------------------------------------------------------------------------------------- |
| **Pasos del eje Z**               | El cabezal va a cada desplazamiento con movimientos Z relativos y vuelve a la altura inicial      |
| **Manual (pausa entre líneas)**   | El trabajo se pausa (`M0`) antes de cada línea; mueves el cabezal a mano y pulsas Reanudar        |
| **Rampa (material inclinado)**    | Una línea larga con marcas de distancia; levantas un extremo de una tira plana                    |

Los pasos del eje Z solo se ofrecen en máquinas con eje Z, y tanto los pasos del eje Z como el
método manual necesitan un controlador G-code, porque usan operaciones de [Comando](command.md)
entre las líneas. Los desplazamientos están limitados a ±10 mm de la altura inicial.

Con el método manual, enfoca primero el láser como de costumbre. Las etiquetas se graban a esa
altura. En la primera pausa, coloca el cabezal en el primer desplazamiento; en cada pausa siguiente,
muévelo un paso. Comprueba que tu controlador se detiene con `M0` y que el botón Reanudar continúa
el trabajo antes de confiar en ello, por ejemplo con una prueba en seco al 0 % de potencia.

En la rampa, la altura bajo cualquier punto de la línea se deduce de la subida de la tira: a una
distancia _d_ a lo largo de una línea de longitud _L_ sobre una tira que sube _h_, el material está
_h_ × _d_ / _L_ más alto que al principio.
