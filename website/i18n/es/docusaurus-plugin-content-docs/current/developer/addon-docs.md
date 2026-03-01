# Desarrollo de Addons

Rayforge usa un sistema de addons basado en [pluggy](https://pluggy.readthedocs.io/) para permitir a desarrolladores extender funcionalidad, añadir nuevos drivers de máquina, o integrar lógica personalizada sin modificar el código base principal.

## 1. Inicio Rápido

La forma más rápida de comenzar es usando la plantilla oficial.

1. **Haz Fork o Clona** la [rayforge-addon-template](https://github.com/barebaric/rayforge-addon-template).
2. **Renombra** el directorio y actualiza los metadatos.

## 2. Estructura del Addon

El `AddonManager` escanea el directorio `addons`. Un addon válido debe ser un directorio que contenga un archivo de manifiesto:

**Estructura de Directorio:**

```text
my-rayforge-addon/
├── rayforge-addon.yaml  <-- Manifiesto Requerido
├── my_addon/            <-- Paquete Python
│   ├── __init__.py
│   ├── backend.py       <-- Punto de entrada backend
│   └── frontend.py      <-- Punto de entrada frontend (opcional)
├── assets/              <-- Recursos opcionales
├── locales/             <-- Traducciones opcionales (archivos .po)
└── README.md
```

## 3. El Manifiesto (`rayforge-addon.yaml`)

Este archivo le dice a Rayforge cómo cargar tu addon.

```yaml
# rayforge-addon.yaml

# Identificador único para tu addon (nombre del directorio)
name: my_custom_addon

# Nombre para mostrar legible por humanos
display_name: "Mi Addon Personalizado"

# Descripción mostrada en la UI
description: "Añade soporte para la cortadora láser XYZ."

# Versión de API (debe coincidir con PLUGIN_API_VERSION de Rayforge)
api_version: 1

# Dependencias de versión de Rayforge
depends:
  - rayforge>=0.27.0,<2.0.0

# Opcional: Dependencias de otros addons
requires:
  - some-other-addon>=1.0.0

# Lo que proporciona el addon
provides:
  # Módulo backend (cargado en procesos principal y worker)
  backend: my_addon.backend
  # Módulo frontend (cargado solo en proceso principal, para UI)
  frontend: my_addon.frontend
  # Archivos de assets opcionales
  assets:
    - path: assets/profiles.json
      type: profiles

# Metadatos del autor
author:
  name: Juan García
  email: juan@example.com

url: https://github.com/username/my-custom-addon
```

### Campos Requeridos

- `name`: Identificador único (debe coincidir con el nombre del directorio)
- `display_name`: Nombre legible mostrado en la UI
- `description`: Descripción breve de la funcionalidad del addon
- `api_version`: Debe ser `1` (coincide con `PLUGIN_API_VERSION` de Rayforge)
- `depends`: Lista de restricciones de versión para Rayforge
- `author`: Objeto con `name` (requerido) y `email` (opcional)

### Campos Opcionales

- `requires`: Lista de dependencias de otros addons
- `provides`: Puntos de entrada y assets
- `url`: Página del proyecto o repositorio

## 4. Puntos de Entrada

Los addons pueden proporcionar dos tipos de puntos de entrada:

### Backend (`provides.backend`)

Cargado tanto en el proceso principal como en los procesos worker. Úsalo para:
- Drivers de máquina
- Tipos de pasos
- Productores de ops
- Funcionalidad principal sin dependencias de UI

### Frontend (`provides.frontend`)

Cargado solo en el proceso principal. Úsalo para:
- Componentes UI
- Widgets GTK
- Elementos de menú
- Acciones que requieren la ventana principal

Los puntos de entrada se especifican como rutas de módulo con puntos (ej., `my_addon.backend`).

## 5. Escribir el Código del Addon

Rayforge usa hooks de `pluggy`. Para conectarte a Rayforge, define funciones decoradas con `@pluggy.HookimplMarker("rayforge")`.

### Boilerplate Básico (`backend.py`)

```python
import logging
import pluggy
from rayforge.context import RayforgeContext

hookimpl = pluggy.HookimplMarker("rayforge")
logger = logging.getLogger(__name__)

@hookimpl
def rayforge_init(context: RayforgeContext):
    """
    Llamado cuando Rayforge está completamente inicializado.
    Este es tu punto de entrada principal para acceder a managers.
    """
    logger.info("¡Mi Addon Personalizado ha iniciado!")

    machine = context.machine
    if machine:
        logger.info(f"Addon ejecutándose en máquina: {machine.id}")

@hookimpl
def on_unload():
    """
    Llamado cuando el addon está siendo deshabilitado o descargado.
    Limpiar recursos, cerrar conexiones, desregistrar handlers.
    """
    logger.info("Mi Addon Personalizado se está cerrando")

@hookimpl
def register_machines(machine_manager):
    """
    Llamado durante el inicio para registrar nuevos drivers de máquina.
    """
    from .my_driver import MyNewMachine
    machine_manager.register("my_new_machine", MyNewMachine)

@hookimpl
def register_steps(step_registry):
    """
    Llamado para registrar tipos de pasos personalizados.
    """
    from .my_step import MyCustomStep
    step_registry.register("my_custom_step", MyCustomStep)

@hookimpl
def register_producers(producer_registry):
    """
    Llamado para registrar productores de ops personalizados.
    """
    from .my_producer import MyProducer
    producer_registry.register("my_producer", MyProducer)

@hookimpl
def register_step_widgets(widget_registry):
    """
    Llamado para registrar widgets de configuración de pasos personalizados.
    """
    from .my_widget import MyStepWidget
    widget_registry.register("my_custom_step", MyStepWidget)

@hookimpl
def register_menu_items(menu_registry):
    """
    Llamado para registrar elementos de menú.
    """
    from .menu_items import register_menus
    register_menus(menu_registry)

@hookimpl
def register_commands(command_registry):
    """
    Llamado para registrar comandos del editor.
    """
    from .commands import register_commands
    register_commands(command_registry)

@hookimpl
def register_actions(window):
    """
    Llamado para registrar acciones de ventana.
    """
    from .actions import setup_actions
    setup_actions(window)
```

### Hooks Disponibles

Definidos en `rayforge/core/hooks.py`:

**`rayforge_init`** (`context`)
: **Punto de Entrada Principal.** Llamado después de que config, camera y hardware están cargados. Úsalo para lógica, inyecciones UI o listeners.

**`on_unload`** ()
: Llamado cuando un addon está siendo deshabilitado o descargado. Úsalo para limpiar
  recursos, cerrar conexiones, desregistrar handlers, etc.

**`register_machines`** (`machine_manager`)
: Llamado durante el inicio para registrar nuevos drivers de máquina.

**`register_steps`** (`step_registry`)
: Llamado para permitir a plugins registrar tipos de pasos personalizados.

**`register_producers`** (`producer_registry`)
: Llamado para permitir a plugins registrar productores de ops personalizados.

**`register_step_widgets`** (`widget_registry`)
: Llamado para permitir a plugins registrar widgets de configuración de pasos personalizados.

**`register_menu_items`** (`menu_registry`)
: Llamado para permitir a plugins registrar elementos de menú.

**`register_commands`** (`command_registry`)
: Llamado para permitir a plugins registrar comandos del editor.

**`register_actions`** (`window`)
: Llamado para permitir a plugins registrar acciones de ventana.

## 6. Acceder a Datos de Rayforge

El hook `rayforge_init` proporciona el **`RayforgeContext`**. A través de este objeto, puedes acceder:

- **`context.machine`**: La instancia de máquina actualmente activa.
- **`context.config`**: Ajustes de configuración global.
- **`context.config_mgr`**: Gestor de configuración.
- **`context.machine_mgr`**: Gestor de máquinas (todas las máquinas).
- **`context.camera_mgr`**: Acceder a feeds de cámara y herramientas de visión por computadora.
- **`context.material_mgr`**: Acceder a la biblioteca de materiales.
- **`context.recipe_mgr`**: Acceder a recetas de procesamiento.
- **`context.dialect_mgr`**: Gestor de dialectos G-code.
- **`context.language`**: Código de idioma actual para contenido localizado.
- **`context.addon_mgr`**: Instancia del gestor de addons.
- **`context.plugin_mgr`**: Instancia del gestor de plugins.
- **`context.debug_dump_manager`**: Gestor de volcados de depuración.
- **`context.artifact_store`**: Almacén de artefactos del pipeline.

## 7. Localización

Los addons pueden proporcionar traducciones usando archivos `.po`:

```text
my-rayforge-addon/
├── locales/
│   ├── de/
│   │   └── LC_MESSAGES/
│   │       └── my_addon.po
│   └── es/
│       └── LC_MESSAGES/
│           └── my_addon.po
```

Los archivos `.po` se compilan automáticamente a archivos `.mo` cuando el addon
se instala o carga.

## 8. Desarrollo y Pruebas

Para probar tu addon localmente sin publicarlo:

1.  **Localiza tu Directorio de Configuración:**
    Rayforge usa `platformdirs`.

    - **Windows:** `C:\Users\<User>\AppData\Local\rayforge\rayforge\addons`
    - **macOS:** `~/Library/Application Support/rayforge/addons`
    - **Linux:** `~/.config/rayforge/addons`
      _(Revisa los logs al inicio para `Config dir is ...`)_

2.  **Crea un symlink a tu addon:**
    En lugar de copiar archivos de un lado a otro, crea un enlace simbólico desde tu carpeta de desarrollo a la carpeta de addons de Rayforge.

    _Linux/macOS:_

    ```bash
    ln -s /path/to/my-rayforge-addon ~/.config/rayforge/addons/my-rayforge-addon
    ```

3.  **Reinicia Rayforge:**
    La aplicación escanea el directorio al inicio. Revisa los logs de consola para:
    > `Loaded addon: my_custom_addon`

## 9. Publicación

Para compartir tu addon con la comunidad:

1.  **Hostea en Git:** Sube tu código a un repositorio Git público (GitHub, GitLab, etc.).
2.  **Envía al Registro:**
    - Ve a [rayforge-registry](https://github.com/barebaric/rayforge-registry).
    - Haz fork del reposorio.
    - Añade la URL Git de tu addon y metadatos a la lista del registro.
    - Envía un Pull Request.

Una vez aceptado, los usuarios pueden instalar tu addon directamente via la UI de Rayforge o usando la URL Git.
