---
description: "Crea copias de arrays con los modos Cuadrícula, Rotación Puntual y Circular. Cada modo ofrece una vista previa en vivo y colocación interactiva."
---

# Arrays

La función de Array te permite crear múltiples copias de piezas de trabajo
seleccionadas usando tres modos de diseño diferentes. Cada modo abre un
diálogo no modal, para que puedas seguir interactuando con el lienzo mientras
ajustas los parámetros — la vista previa se actualiza en tiempo real.

Para abrir un diálogo de array, selecciona una o más piezas de trabajo en el
lienzo, luego elige el modo de array desde la barra de herramientas o el
menú contextual.

:::tip
Todos los modos de array son no modales. Puedes arrastrar piezas de trabajo
en el lienzo mientras el diálogo está abierto, y la vista previa se
actualizará en vivo para reflejar las nuevas posiciones.
:::

---

## Cuadrícula

El modo Cuadrícula organiza las copias en una matriz rectangular de filas y
columnas, con espaciado horizontal y vertical configurable.

![Array Cuadrícula](/screenshots/main-array-grid.png)

### Configuración

| Parámetro | Descripción |
|-----------|-------------|
| **Filas** | Número de filas (1–360) |
| **Columnas** | Número de columnas (1–360) |
| **Modo de espaciado** | Elige entre *Hueco* (espacio entre copias) o *Passo* (distancia de borde a borde de cada copia) |
| **Espaciado de columnas** | Espaciado horizontal entre columnas |
| **Espaciado de filas** | Espaciado vertical entre filas |

---

## Rotación Puntual

El modo Rotación Puntual crea copias rotándolas en su lugar alrededor del
propio centro de la selección. Esto es útil para crear patrones circulares
donde cada copia permanece en su ubicación original pero se rota por una
fracción del ángulo total.

![Array Rotación Puntual](/screenshots/main-array-point-rotation.png)

### Configuración

| Parámetro | Descripción |
|-----------|-------------|
| **Cantidad** | Número de copias (1–360) |
| **Ángulo total (grados)** | Extensión angular total de todas las copias (−360° a 360°) |

:::info
Dado que la rotación es alrededor del propio centro de la selección,
arrastrar la pieza de trabajo en el lienzo mueve todas las copias juntas
mientras el diálogo permanece abierto.
:::

---

## Circular

El modo Circular coloca copias a lo largo de un arco circular alrededor de
un punto central. Un marcador de cruz en el lienzo muestra el centro, y
puedes arrastrarlo a una nueva posición mientras el diálogo está abierto.

![Array Circular](/screenshots/main-array-circular.png)

### Configuración

| Parámetro | Descripción |
|-----------|-------------|
| **Cantidad** | Número de copias (1–360) |
| **Ángulo total (grados)** | Extensión angular del arco (−360° a 360°) |
| **Centro X** | Coordenada X del centro del círculo |
| **Centro Y** | Coordenada Y del centro del círculo |
| **Radio** | Radio de la trayectoria circular |
| **Rotar copias** | Cuando está habilitado, cada copia se rota para seguir la tangente del arco |

:::tip Arrastrar el centro
La cruz en el lienzo representa el centro del círculo. Arrástrala para
reposicionar el array interactivamente — los campos Centro X y Centro Y en
el diálogo se actualizarán automáticamente.
:::

:::tip Arrastrar piezas de trabajo
También puedes arrastrar la pieza de trabajo original en el lienzo. El radio
se actualizará automáticamente para mantener las copias a su distancia actual
del centro.
:::
