# Guía de Desarrollo de Paquetes Rayforge

Rayforge usa un sistema de paquetes basado en [pluggy](https://pluggy.readthedocs.io/) para permitir a desarrolladores extender funcionalidad, añadir nuevos drivers de máquina, o integrar lógica personalizada sin modificar el código base principal.

## 1. Inicio Rápido

La forma más rápida de comenzar es usando la plantilla oficial.

1. **Haz Fork o Clona** la [rayforge-package-template](https://github.com/barebaric/rayforge-package-template).
2. **Renombra** el directorio y actualiza los metadatos.

## 2. Estructura del Paquete

El `PackageManager` escanea el directorio `packages`. Un paquete válido debe ser un directorio que contenga al menos dos archivos:

1. `rayforge_package.yaml` (Metadatos)
2. Un punto de entrada Python (ej., `package.py`)

**Estructura de Directorio:**

```text
my-rayforge-package/
├── rayforge_package.yaml  <-- Manifiesto Requerido
├── package.py             <-- Punto de entrada (lógica)
├── assets/                <-- Recursos opcionales
└── README.md
```

## 3. El Manifiesto (`rayforge_package.yaml`)

Este archivo le dice a Rayforge cómo cargar tu paquete.

```yaml
# rayforge_package.yaml

# Identificador único para tu paquete
name: my_custom_package

# Nombre para mostrar legible por humanos
display_name: "Mi Paquete Personalizado"

# Cadena de versión
version: 0.1.0

# Descripción mostrada en la UI
description: "Añade soporte para la cortadora láser XYZ."

# Dependencias (paquete y restricciones de versión)
depends:
  - rayforge>=0.27.0,~0.27

# El archivo python a cargar (relativo a la carpeta del paquete)
entry_point: package.py

# Metadatos del autor
author: Jane Doe
url: https://github.com/username/my-custom-package
```

## 4. Escribir el Código del Paquete

Rayforge usa hooks de `pluggy`. Para conectarte a Rayforge, define funciones decoradas con `@pluggy.HookimplMarker("rayforge")`.

### Boilerplate Básico (`package.py`)

```python
import logging
import pluggy
from rayforge.context import RayforgeContext

# Define el marcador de implementación de hook
hookimpl = pluggy.HookimplMarker("rayforge")
logger = logging.getLogger(__name__)

@hookimpl
def rayforge_init(context: RayforgeContext):
    """
    Llamado cuando Rayforge está completamente inicializado.
    Este es tu punto de entrada principal para acceder a managers.
    """
    logger.info("¡Mi Paquete Personalizado ha iniciado!")

    # Accede a sistemas core via el contexto
    machine = context.machine
    camera = context.camera_mgr

    if machine:
        logger.info(f"Paquete ejecutándose en máquina: {machine.id}")

@hookimpl
def register_machines(machine_manager):
    """
    Llamado durante el inicio para registrar nuevos drivers de máquina.
    """
    # from .my_driver import MyNewMachine
    # machine_manager.register("my_new_machine", MyNewMachine)
    pass
```

### Hooks Disponibles

Definidos en `rayforge/core/hooks.py`:

**`rayforge_init`** (`context`)
: **Punto de Entrada Principal.** Llamado después de que config, camera y hardware están cargados. Úsalo para lógica, inyecciones UI o listeners.

**`register_machines`** (`machine_manager`)
: Llamado temprano en el proceso de arranque. Úsalo para registrar nuevas clases/drivers de hardware.

## 5. Acceder a Datos de Rayforge

El hook `rayforge_init` proporciona el **`RayforgeContext`**. A través de este objeto, puedes acceder:

- **`context.machine`**: La instancia de máquina actualmente activa.
- **`context.config`**: Ajustes de configuración global.
- **`context.camera_mgr`**: Acceder a feeds de cámara y herramientas de visión por computadora.
- **`context.material_mgr`**: Acceder a la biblioteca de materiales.
- **`context.recipe_mgr`**: Acceder a recetas de procesamiento.

## 6. Desarrollo y Pruebas

Para probar tu paquete localmente sin publicarlo:

1.  **Localiza tu Directorio de Configuración:**
    Rayforge usa `platformdirs`.

    - **Windows:** `C:\Users\<User>\AppData\Local\rayforge\rayforge\packages`
    - **macOS:** `~/Library/Application Support/rayforge/packages`
    - **Linux:** `~/.config/rayforge/packages`
      _(Revisa los logs al inicio para `Config dir is ...`)_

2.  **Crea un symlink a tu paquete:**
    En lugar de copiar archivos de un lado a otro, crea un enlace simbólico desde tu carpeta de desarrollo a la carpeta de paquetes de Rayforge.

    _Linux/macOS:_

    ```bash
    ln -s /path/to/my-rayforge-package ~/.config/rayforge/packages/my-rayforge-package
    ```

3.  **Reinicia Rayforge:**
    La aplicación escanea el directorio al inicio. Revisa los logs de consola para:
    > `Loaded package: my_custom_package`

## 7. Publicación

Para compartir tu paquete con la comunidad:

1.  **Hostea en Git:** Sube tu código a un repositorio Git público (GitHub, GitLab, etc.).
2.  **Envía al Registro:**
    - Ve a [rayforge-registry](https://github.com/barebaric/rayforge-registry).
    - Haz fork del repositorio.
    - Añade la URL Git de tu paquete y metadatos a la lista del registro.
    - Envía un Pull Request.

Una vez aceptado, los usuarios pueden instalar tu paquete directamente via la UI de Rayforge o usando la URL Git.
