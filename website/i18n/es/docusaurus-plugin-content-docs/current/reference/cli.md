---
description: "Referencia de la línea de comandos de Rayforge."
---

# Línea de Comandos

Referencia completa de las opciones de línea de comandos de Rayforge.

```
rayforge [opciones] [archivos...]
```

---

## Argumentos posicionales

| Argumento  | Descripción                       |
| ---------- | --------------------------------- |
| `archivos` | Archivos SVG o imagen al iniciar. |

---

## Opciones

| Opción              | Descripción                                |
| ------------------- | ------------------------------------------ |
| `--version`         | Imprimir versión y salir.                  |
| `-h`, `--help`      | Mostrar ayuda y salir.                     |
| `--loglevel NIVEL`  | Nivel de registro. Por defecto: `INFO`.    |
| `--config DIR`      | Directorio de configuración personalizado. |
| `--exit`            | Salir después del importe.                 |
| `--vector`          | Forzar importación como vectores directos. |
| `--trace`           | Forzar importación por trazado de bitmap.  |
| `--script SCRIPT`   | Script de inicio temprano.                 |
| `--uiscript SCRIPT` | Script de UI (post-carga).                 |

---

## Ejemplos

### Abrir un archivo

```bash
rayforge miproyecto.ryp
```

### Abrir varios archivos

```bash
rayforge pieza1.svg logo.png diseno.ryp
```

### Importar con trazado

```bash
rayforge --trace foto.png
```

### Ejecutar script temprano y salir

```bash
rayforge --exit --script registrar.py \
    miproyecto.ryp
```

### Script de UI (automatización)

```bash
rayforge --exit --uiscript screenshot.py \
    miproyecto.ryp
```

### Procesamiento por lotes

```bash
rayforge --exit --vector entrada.svg
```

---

## Scripts tempranos (`--script`)

El flag `--script` ejecuta un script de Python **sincrónicamente
durante el inicio**, antes de que se carguen los complementos y
antes de que se cree la ventana principal. Útil para:

- Registrar complementos con el gestor de complementos `pluggy`
- Configurar el contexto de la aplicación
- Registrar funciones de plantilla para cuadros de texto
- Establecer variables de entorno antes del inicio

El script tiene acceso al contexto vía `get_context()`:

```python
from rayforge.context import get_context

ctx = get_context()
# Registrar complementos, configurar servicios, etc.
```

### Ejemplo: Registrar una función de plantilla personalizada

```python
"""Registrar función personalizada para plantillas de texto.

Ejecutar con: rayforge --script registrar_fn.py
"""
from rayforge.context import get_context
from sketcher.core.template_functions import (
    register_template_function,
)

register_template_function("mi_id", lambda: "PARTE-001")
```

Ahora `{mi_id()}` funciona en cualquier cuadro de texto.

Consulte
[Funciones de plantilla personalizadas](../features/sketcher.md#custom-template-functions)
en la documentación del sketcher para un tutorial completo.

---

## Scripts de UI (`--uiscript`)

El flag `--uiscript` ejecuta un script de Python **después de
que la ventana principal se carga completamente**, en un hilo en
segundo plano. Útil para:

- Pruebas automatizadas de UI
- Capturar pantallas de la aplicación
- Flujos de trabajo de extremo a extremo

El script puede importar la aplicación y la ventana directamente:

```python
from rayforge.uiscript import app, win
```

El script se ejecuta en un **hilo en segundo plano** — tenga en
cuenta la seguridad de hilos al acceder a widgets GTK
(use `GLib.idle_add` para operaciones GTK).

### Ejemplo: Capturar pantalla

```python
"""Capturar pantalla de la ventana principal."""
from rayforge.uiscript import app, win

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import GLib

def capture():
    surface = win.get_surface()
    if surface:
        surface.write_to_png("/tmp/rayforge_screenshot.png")
    return GLib.SOURCE_REMOVE

GLib.idle_add(capture)
```

---

## Usar ambos flags

Ambos `--script` y `--uiscript` se pueden usar juntos.
El `--script` se ejecuta primero (sincrónicamente), luego se
carga la ventana, y luego se ejecuta `--uiscript`:

```bash
rayforge --script setup_temprano.py \
    --uiscript automatizacion.py \
    miproyecto.ryp
```

Esto es útil cuando necesita registrar complementos primero
y luego controlar la UI más tarde.
