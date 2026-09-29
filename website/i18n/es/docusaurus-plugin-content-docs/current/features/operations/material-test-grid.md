---
description:
  "Genera una cuadrícula de prueba de material para encontrar los ajustes óptimos de potencia y
  velocidad del láser en cualquier material. Calibra tu cortadora láser de forma sistemática."
---

# Cuadrícula de Prueba de Material

Cada material — y a menudo cada color y espesor del mismo material — responde de forma diferente a
la potencia y velocidad del láser. La Cuadrícula de Prueba de Material elimina las conjeturas al
buscar la combinación adecuada: genera un patrón de celdas de prueba en el que cada celda se graba o
corta con un ajuste ligeramente diferente, todo en un solo trabajo. Después de una ejecución podrás
ver de un vistazo qué combinación produce el resultado que deseas.

Crea una mediante **Herramientas → Crear Cuadrícula de Prueba de Material**. Rayforge añade una
pieza de trabajo especial al lienzo junto con una operación correspondiente, y configuras la
cuadrícula en su diálogo de ajustes.

![Ajustes de Cuadrícula de Prueba de Material](/screenshots/material-test.webp)

## Preajustes

El diálogo de ajustes ofrece preajustes para tipos comunes de láser. Rellenan un rango de velocidad
sensato, un rango de potencia y el tipo de prueba para que puedas empezar con una base razonable:

| Preajuste         | Rango de Velocidad | Rango de Potencia | Tipo de Prueba |
| ----------------- | ------------------ | ----------------- | -------------- |
| **Grabado Diodo** | 1000-10000 mm/min  | 10-100%           | Grabar         |
| **Corte Diodo**   | 100-5000 mm/min    | 50-100%           | Cortar         |
| **Grabado CO2**   | 3000-20000 mm/min  | 10-50%            | Grabar         |
| **Corte CO2**     | 1000-20000 mm/min  | 30-100%           | Cortar         |

Un preajuste es solo un punto de partida — todos los valores siguen siendo ajustables después, y los
rangos de velocidad se limitan automáticamente a lo que tu máquina puede hacer.

## Modos de Cuadrícula

Una cuadrícula de prueba varía dos parámetros a la vez: uno a lo largo de las columnas y otro a lo
largo de las filas. El modo de cuadrícula decide cuáles son. **Potencia vs Velocidad** es el modo
predeterminado y cubre la pregunta más común — potencia en las columnas, velocidad en las filas.

**Potencia vs Pasadas** y **Velocidad vs Pasadas** mantienen fijo uno de los dos y varían el número
de pasadas en su lugar, lo cual es útil para cortar material grueso. **Velocidad vs Desfase** es un
modo de calibración especial para grabado bidireccional: varía el desfase horizontal de escaneo para
que puedas corregir la desalineación entre filas. Como eso solo tiene sentido para trabajos raster,
al seleccionarlo la cuadrícula cambia a Grabar y se amplía el espaciado de líneas para que cualquier
desalineación sea fácil de ver. Dentro de cada fila la potencia se escala junto con la velocidad, de
modo que todas las celdas permanezcan visualmente comparables.

## Configurar la Cuadrícula

El diálogo de ajustes agrupa los parámetros en tres secciones.

La sección **Cuadrícula** controla la prueba en sí. El tipo de prueba determina si cada celda corta
el contorno de un cuadrado o lo rellena con líneas raster. Las dimensiones de la cuadrícula
establecen cuántas columnas y filas probar — cada columna representa un paso del primer parámetro
del modo y cada fila un paso del segundo, desde el mínimo hasta el máximo del rango que introduzcas.
Se permiten entre 2 y 20 pasos por eje; 5×5 es un buen valor predeterminado. El tamaño de forma (10
mm por defecto) y el espaciado (2 mm por defecto) determinan cuán grande se vuelve la cuadrícula.
Para el tipo de prueba Grabar, el intervalo de línea controla la distancia entre las líneas de
escaneo — los valores más pequeños rellenan de forma más densa pero tardan más. Déjalo en cero para
usar el tamaño del punto de tu láser, que es una buena opción para la mayoría de los grabados.

La sección **Etiquetas** controla las anotaciones grabadas junto a la cuadrícula. Las etiquetas
están activadas por defecto y se graban primero, de modo que el patrón de prueba no pueda
ocultarlas. Tienen su propia potencia (10% por defecto) y velocidad (1000 mm/min por defecto), y los
valores de velocidad se muestran en tu unidad de visualización preferida.

La sección **Parámetros** contiene los rangos que varía la cuadrícula — velocidad, potencia, pasadas
o desfase, según el modo seleccionado. Los modos que mantienen un parámetro fijo (por ejemplo la
velocidad en Potencia vs Pasadas) también te permiten establecer esa constante aquí.

## Entendiendo el Diseño

En el modo predeterminado Potencia vs Velocidad, la potencia aumenta de izquierda a derecha y la
velocidad de arriba abajo:

```
                   Potencia (%)
                 10       55       100
Velocidad  100  [  ]     [  ]     [  ]
(mm/min)   300  [  ]     [  ]     [  ]
           500  [  ]     [  ]     [  ]
```

Las etiquetas en los bordes izquierdo y superior muestran el valor exacto de cada fila y columna, de
modo que nunca tengas que contar celdas.

El tamaño total se deduce directamente de las dimensiones de la cuadrícula: cada eje mide _pasos ×
tamaño de forma + (pasos − 1) × espaciado_, más el espacio para las etiquetas a la izquierda y
arriba (como máximo 15 mm, y solo cuando las etiquetas están activadas). Una cuadrícula de 5×5 con
cuadrados de 20 mm y espaciado de 5 mm mide 120 mm por lado sin etiquetas y 135 mm con ellas.

## Cómo se Ejecuta la Cuadrícula

Las celdas deliberadamente **no** se ejecutan en orden de lectura. Rayforge las ejecuta en un orden
optimizado por riesgo: primero la velocidad más alta, la potencia más baja dentro de cada velocidad
y el menor número de pasadas dentro de cada potencia. Las combinaciones lentas y de alta potencia
son las más propensas a chamuscar el material o iniciar un fuego, por lo que se ejecutan al final.
Este orden es intencional y no puede cambiarse.

## Ejecutar la Prueba

Carga el material que quieres caracterizar — material de desecho, no tu pieza final — y enfoca el
láser como lo harías para un trabajo real, ya que la distancia de enfoque cambia el resultado.
Inicia el trabajo y quédate junto a la máquina: si una celda comienza a chamuscarse gravemente o a
echar demasiado humo, detén el trabajo en lugar de dejarlo terminar.

Cuando la prueba termine, examina cada celda. Si el grabado sale demasiado claro, muévete hacia más
potencia o menor velocidad; si sale oscuro o chamuscado, muévete hacia menos potencia o mayor
velocidad. Para las pruebas de corte, busca la celda que corta completamente con la menor chamusca.
Para acercarte al punto óptimo, ejecuta una segunda cuadrícula más fina: si una prueba gruesa de 5×5
encontró su mejor celda alrededor del 40% de potencia y 4000 mm/min, una cuadrícula de seguimiento
que abarque 35-45% y 3000-5000 mm/min la ubicará con precisión.

<!-- prettier-ignore-start -->
:::tip[Guárdalo como receta]
En lugar de llevar un cuaderno con los ajustes ganadores, guárdalos como una
[receta](../../application-settings/recipes.md): ponle nombre (por ejemplo "Corte de Contrachapado
de 3 mm"), vincúlala a la máquina, la operación, el material y el espesor que probaste, y Rayforge
sugerirá exactamente esos ajustes la próxima vez que cortes el mismo material.
:::
<!-- prettier-ignore-end -->

## Uso Avanzado

Las cuadrículas de prueba de material son piezas de trabajo normales, por lo que se combinan
libremente con otras operaciones. Un patrón común es añadir una operación de contorno alrededor de
la cuadrícula terminada y cortar la pieza de prueba del material base cuando el grabado termina.

Ejecutar la misma configuración de cuadrícula en diferentes materiales es una forma rápida de
construir una biblioteca de ajustes confiables — y las recetas hacen que esa biblioteca se pueda
buscar por material y espesor más adelante.

## Consejos y Mejores Prácticas

Algunos hábitos hacen que los resultados de las pruebas sean más confiables:

- Comienza desde un preajuste y ajusta a partir de ahí en lugar de configurar desde cero.
- Dale espacio a las celdas: los cuadrados de 15-20 mm son mucho más fáciles de evaluar que los
  diminutos.
- Cambia una variable a la vez al acotar resultados — una cuadrícula fina que varía ambos ejes en un
  rango amplio es difícil de interpretar.
- Deja que el material se enfríe entre pruebas consecutivas en la misma pieza.
- Usa la misma distancia de enfoque en cada prueba, incluido el trabajo final.

Y las reglas habituales de seguridad láser se aplican doble a las cuadrículas de prueba, que
exploran intencionadamente territorio desconocido:

- Nunca dejes una prueba en ejecución sin supervisión.
- Comienza con rangos de potencia conservadores y ve subiendo.
- Asegúrate de que la extracción de humos funcione antes de comenzar.
- Ten un extintor al alcance de la mano.

## Solución de Problemas

**Las celdas se ejecutan en un orden extraño.** Ese es el orden de ejecución optimizado por riesgo
descrito en [Cómo se Ejecuta la Cuadrícula](#how-the-grid-runs) — primero las combinaciones más
rápidas y de menor potencia. Es intencional.

**Los resultados son inconsistentes entre ejecuciones.** Asegúrate de que el material esté plano y
sujeto, de que el enfoque sea idéntico en toda la cuadrícula y de que tu fuente de alimentación
entregue una potencia estable. Si solo una región de la cuadrícula se ve mal, el propio material
puede ser irregular.

## Temas Relacionados

- **[Vista Previa 3D](../../ui/3d-preview.md)** - Previsualiza la ejecución de la prueba antes de
  ejecutarla
- **[Recetas](../../application-settings/recipes.md)** - Reutiliza tus resultados de prueba
  automáticamente
- **[Grabado](engrave)** - Entender las operaciones de grabado
- **[Corte de Contorno](contour)** - Entender las operaciones de corte
