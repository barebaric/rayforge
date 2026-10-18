# Optimización de Trayectoria

La optimización de trayectoria reordena los segmentos de corte para minimizar la distancia de viaje.
El láser se mueve eficientemente entre cortes en lugar de saltar aleatoriamente a través del área de
trabajo.

## Cómo Funciona

Sin optimización, las trayectorias se cortan en el orden en que aparecen en tu archivo de diseño. La
optimización analiza todos los segmentos de trayectoria y los reorganiza para que el láser viaje la
distancia total más corta entre cortes.

**Antes de la optimización** — el láser salta de un lado a otro a través del material:

![Los desplazamientos saltan entre cortes en el orden dibujado](/images/path-optimization-before.webp)

**Después de la optimización** — el láser se mueve secuencialmente de corte a corte:

![Los desplazamientos conectan cortes cercanos](/images/path-optimization-after.webp)

## Ajustes

![Los ajustes de Optimización en la página de posprocesamiento de un paso de contorno](/screenshots/step-settings-contour-post-optimize.webp)

### Habilitar Optimización

Activa o desactiva la optimización de trayectoria. Habilitada por defecto para la mayoría de
operaciones.

### Elegir el Mejor Punto de Inicio

Inicia cada trayectoria cerrada en el vértice más cercano al cabezal láser en lugar de donde se
dibujó. Esto acorta los desplazamientos en vacío. Las trayectorias abiertas y las que tienen
entradas, sobrecorte o pestañas de sujeción conservan su punto de inicio. Desactivado por defecto.

### Preferir Esquinas

Solo disponible cuando **Elegir el Mejor Punto de Inicio** está activado. Las trayectorias cerradas
empiezan en una esquina aguda si la tienen, lo que hace que la costura sea menos visible. Las
trayectorias sin esquinas, como los círculos, empiezan en el vértice más cercano. Desactivado por
defecto.

La misma trayectoria cerrada iniciada donde se dibujó (1), en el vértice más cercano al cabezal
láser con **Elegir el Mejor Punto de Inicio** (2), y en la esquina aguda más cercana con **Preferir
Esquinas** (3):

![Puntos de inicio de una trayectoria cerrada: como se dibujó, en el vértice más cercano, en la
esquina más cercana](/images/path-optimization-start-point.webp)

## Cuándo Usar Optimización

**Habilitar para:**

- Diseños con muchas formas separadas
- Reducir el tiempo total del trabajo
- Minimizar el desgaste del sistema de movimiento
- Diseños anidados complejos

**Deshabilitar para:**

- Diseños donde el orden de corte importa (ej., características interiores antes que exteriores)
- Depuración de problemas de trayectoria
- Cuando necesitas un orden de ejecución predecible y repetible

## Cómo Afecta Tu Trabajo

**Ahorro de tiempo:** Puede reducir el tiempo de trabajo en 20-50% para diseños con muchos cortes
separados.

**Eficiencia de movimiento:** Menos movimiento rápido significa menos desgaste en correas, motores y
rodamientos.

**Distribución de calor:** Las trayectorias optimizadas pueden concentrar calor en un área. Para
materiales sensibles al calor, considera si el orden importa.

<!-- prettier-ignore-start -->
:::tip
La optimización se ejecuta automáticamente. Simplemente habilítala y el software maneja el
resto.
:::
<!-- prettier-ignore-end -->

---

## Temas Relacionados

- [Corte de Contorno](operations/contour) - Operación de corte principal
- [Pestañas de Sujeción](holding-tabs) - Mantener piezas aseguradas durante el corte
