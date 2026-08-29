---
description: "Plantillas de cuadros de texto y funciones de plantilla personalizadas en el diseñador paramétrico 2D de Rayforge."
---

# Plantillas de texto

Los cuadros de texto soportan expresiones de plantilla entre llaves. Estas se
resuelven en el momento de la resolución usando los valores actuales de los
parámetros, por lo que el texto se actualiza automáticamente al cambiar una
dimensión o variable de entrada.

## Sustitución de variables

Referencia cualquier parámetro del boceto o variable de entrada por nombre:

- `{width}` — el valor actual del parámetro "width"
- `{name}` — el valor de un parámetro de entrada de tipo cadena
- `{count:.0f}` — formateado con un especificador de formato Python (sin decimales)

## Expresiones matemáticas

Puedes usar funciones matemáticas en las plantillas:

- `{sqrt(area):.2f}` — raíz cuadrada de "area", formateada a 2 decimales
- `{width * 2}` — expresiones aritméticas

Las funciones matemáticas estándar (`sqrt`, `sin`, `cos`, `tan`, `pi`, etc.)
están disponibles.

## Funciones integradas

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

## Especificaciones de formato

Las especificaciones de formato de Python funcionan con
cualquier resultado de expresión:

- `{width:.1f}` — un decimal
- `{timestamp():.0f}` — sin decimales en la marca de tiempo
- `{today()}` — representación de cadena predeterminada

## Ejemplos de uso

- `Pieza #{uuid4()}` — número de serie único en cada resolución
- `A={width:.1f} A={height:.1f}` — etiquetas de dimensiones en vivo
- `Fecha: {today()}` — fechar cada pieza
- `{name} - {count:.0f}uds` — combinar parámetros de cadena y numéricos
- `{timestamp():.0f}` — marca de tiempo Unix para registro de producción

## Funciones de plantilla personalizadas

Puedes registrar tus propias funciones para usar dentro de
plantillas de texto. Esto es útil para obtener números de
serie de una base de datos, leer datos externos o generar
etiquetas personalizadas.

### Escribir el script de registro

Crea un archivo Python (ej.
`~/.config/rayforge/mis_funciones.py`):

```python
"""Registrar funciones personalizadas para plantillas."""
import sqlite3

from sketcher.core.template_functions import (
    register_template_function,
)

RUTA_DB = "/home/usuario/produccion.db"


def siguiente_serie() -> str:
    """Obtener siguiente número de serie de la base."""
    conn = sqlite3.connect(RUTA_DB)
    try:
        cur = conn.execute(
            "UPDATE contadores SET valor = valor + 1 "
            "WHERE nombre = 'serial' RETURNING valor"
        )
        row = cur.fetchone()
        conn.commit()
        return f"SN-{row[0]:06d}"
    finally:
        conn.close()


register_template_function("siguiente_serie", siguiente_serie)
```

Puntos clave:

- Llama `register_template_function(nombre, callable)` para
  cada función.
- Tu función puede hacer cualquier cosa que Python pueda:
  abrir archivos, conectarse a bases de datos, llamar APIs,
  etc.
- La función se ejecuta en **cada renderizado**, por lo que
  debe ser rápida.
- Las funciones son seguras en hilos si tu callable lo es.

### Ejecutar Rayforge con el script

Usa el flag `--script` para cargar tus funciones antes de
que se abra la ventana:

```bash
rayforge --script ~/.config/rayforge/mis_funciones.py \
    mi_documento.ryp
```

Esto ejecuta tu script temprano durante el inicio — antes
de que se carguen los complementos y antes de que se cree
la ventana principal — para que la función esté disponible
cuando se resuelve el boceto por primera vez.

### Usar la función en un cuadro de texto

Crea un cuadro de texto con:

```
{siguiente_serie()}
```

Las especificaciones de formato también funcionan:

```
{siguiente_serie():>20}
```

### Registrar funciones programáticamente

Si estás escribiendo un complemento o una biblioteca
reutilizable, puedes llamar a `register_template_function`
desde cualquier código Python que se ejecute antes de que
se resuelva el boceto:

```python
from sketcher.core.template_functions import (
    register_template_function,
)

register_template_function(
    "numero_parte",
    lambda: f"P-{hash('x') % 10000:04d}"
)
```

### Las funciones integradas no se pueden eliminar

Las funciones integradas (`today`, `now`, `uuid`, etc.)
no se pueden eliminar del registro. Si necesitas cambiar
su comportamiento, registra una función con un nombre
diferente.
