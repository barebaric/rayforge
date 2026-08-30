---
description:
  "Parámetros del boceto, expresiones en restricciones y plantillas de cuadros de texto en el
  diseñador de Rayforge: geometría y etiquetas gobernadas por valores y fórmulas con nombre."
---

# Expresiones y parámetros

Un boceto se vuelve verdaderamente paramétrico cuando sus dimensiones están gobernadas por valores
con nombre en lugar de números fijados en el código. El diseñador admite esto en dos lugares: las
restricciones dimensionales aceptan **expresiones**, y los cuadros de texto aceptan **expresiones de
plantilla**. Ambas se evalúan mediante el resolvedor, por lo que el boceto se actualiza
automáticamente cada vez que cambia un valor.

## Parámetros del boceto

Cada boceto lleva su propia lista de parámetros, mostrada en el panel **Parámetros del boceto** a la
izquierda del editor de bocetos. **Añadir parámetro** crea uno, con elección entre un entero, un
número de punto flotante, un deslizador o una sola línea de texto. Cada parámetro tiene un nombre —
la columna `key` — y ese nombre es al que se refieren las expresiones.

Una configuración típica para una caja con grosor de pared variable son dos parámetros, `width` y
`thickness`. Todavía nada restringe la geometría; los parámetros son solo nombres para números hasta
que una expresión los usa.

## Expresiones en restricciones

Haga doble clic en una restricción dimensional (vea [Restricciones](constraints.md)) e introduzca
una expresión en lugar de un número simple:

```
width / 2
```

El valor de la restricción pasa a ser el resultado de esa expresión, reevaluado cada vez que el
boceto se resuelve. Cambie el parámetro `width` y la geometría restringida seguirá el cambio: una
sola edición actualiza cada dimensión que haga referencia a él. Las restricciones gobernadas por una
expresión dibujan su marcador en naranja, y la etiqueta muestra el valor calculado.

Las expresiones pueden combinar parámetros con aritmética y las funciones matemáticas estándar de
Python:

```
width - 2 * thickness
sqrt(area) / 2
2 * pi * radius
```

Funciones como `sqrt`, `sin`, `cos` y `tan`, y constantes como `pi`, provienen del módulo `math` de
Python; ese módulo, más los parámetros, es exactamente lo que una expresión de restricción puede
referenciar. Los parámetros de cadena también pueden referenciarse, lo cual es sobre todo útil en
los cuadros de texto.

## Expresiones de plantilla en cuadros de texto {#template-expressions-in-text-boxes}

Los cuadros de texto resuelven las expresiones entre llaves en el momento de la resolución, de modo
que las etiquetas y el texto grabado muestran valores en vivo:

```
W = {width}, H = {height}
```

Cualquier parámetro puede sustituirse por su nombre, y el resultado puede formatearse con un
especificador de formato de Python tras dos puntos:

- `{width}` — el valor actual del parámetro "width"
- `{name}` — el valor de un parámetro de tipo cadena
- `{width:.1f}` — un decimal
- `{timestamp():.0f}` — sin decimales en el resultado de una función

Aquí también funcionan las matemáticas, ya sea como una expresión como `{width * 2}` o mediante una
función como `{sqrt(area):.2f}`. En comparación con las expresiones de restricción, las plantillas
de texto tienen una caja de herramientas más rica: junto con el módulo matemático exponen las
funciones integradas de abajo, y se pueden registrar funciones personalizadas para ellas (vea
[más abajo](#custom-template-functions)).

### Funciones de plantilla integradas

| Función         | Tipo retorno | Descripción                                          |
| --------------- | ------------ | ---------------------------------------------------- |
| `{today()}`     | `date`       | Fecha UTC actual (ej.: `2026-08-26`)                 |
| `{date()}`      | `date`       | Alias de `today()`                                   |
| `{now()}`       | `datetime`   | Fecha y hora UTC actuales                            |
| `{time()}`      | `time`       | Hora UTC actual (ej.: `15:30:00.123456+00:00`)       |
| `{timestamp()}` | `float`      | Marca de tiempo Unix (segundos desde época)          |
| `{uuid4()}`     | `str`        | Cadena hexadecimal de 8 caracteres (ej.: `a1b2c3d4`) |
| `{uuid8()}`     | `str`        | Alias de `uuid4()`                                   |
| `{uuid()}`      | `str`        | Cadena UUID v4 completa (36 caracteres)              |

Entre los usos típicos se incluyen números de serie únicos en cada resolución (`Pieza #{uuid4()}`),
etiquetas de dimensiones en vivo (`W={width:.1f} H={height:.1f}`), fechas en cada pieza
(`Fecha: {today()}`), contadores de producción (`{name} - {count:.0f}uds`) o marcas de tiempo Unix
para el registro de producción (`{timestamp():.0f}`).

## Funciones de plantilla personalizadas {#custom-template-functions}

Puede registrar sus propias funciones para usar dentro de las plantillas de los cuadros de texto.
Esto es útil para obtener números de serie de una base de datos, leer datos externos o generar
etiquetas personalizadas.

### Escribir el script de registro

Cree un archivo Python (ej. `~/.config/rayforge/my_functions.py`):

```python
"""Register custom template functions for text box expressions."""
import sqlite3

from sketcher.core.template_functions import register_template_function

DB_PATH = "/home/you/production.db"


def next_serial() -> str:
    """Fetch and reserve the next serial number from the database."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.execute(
            "UPDATE counters SET value = value + 1 "
            "WHERE name = 'serial' RETURNING value"
        )
        row = cur.fetchone()
        conn.commit()
        return f"SN-{row[0]:06d}"
    finally:
        conn.close()

register_template_function("next_serial", next_serial)
```

Llame a `register_template_function(name, callable)` para cada función. La función puede hacer
cualquier cosa que Python pueda — abrir archivos, conectarse a bases de datos, llamar APIs — y se
ejecuta en **cada renderizado**, por lo que debe ser rápida (use caché si los datos subyacentes no
cambian entre renderizados). Las funciones son seguras en hilos si su callable lo es.

### Ejecutar Rayforge con el script

Use el flag `--script` para cargar sus funciones antes de que se abra la ventana:

```bash
rayforge --script ~/.config/rayforge/my_functions.py mydoc.ryp
```

Esto ejecuta su script temprano durante el inicio — antes de que se carguen los complementos y antes
de que se cree la ventana principal — para que la función esté disponible cuando el boceto se
resuelva por primera vez.

### Usar la función en un cuadro de texto

En el diseñador, cree un cuadro de texto con:

```
{next_serial()}
```

Las especificaciones de formato también funcionan:

```
{next_serial():>20}
```

### Registrar funciones programáticamente

Si está escribiendo un complemento o una biblioteca reutilizable, puede llamar a
`register_template_function` desde cualquier código Python que se ejecute antes de que se resuelva
el boceto:

```python
from sketcher.core.template_functions import register_template_function

register_template_function("part_number", lambda: f"P-{hash('x') % 10000:04d}")
```

### Las funciones integradas no se pueden eliminar

Las funciones integradas (`today`, `now`, `uuid`, etc.) no se pueden eliminar del registro. Si
necesita cambiar su comportamiento, registre una función con un nombre diferente.
