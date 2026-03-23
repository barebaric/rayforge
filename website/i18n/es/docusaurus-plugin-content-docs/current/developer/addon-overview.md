# Resumen del Desarrollo de Addons

Rayforge utiliza un sistema de addons basado en [pluggy](https://pluggy.readthedocs.io/) que te permite extender la funcionalidad, agregar nuevos controladores de máquinas o integrar lógica personalizada sin modificar el código base principal.

## Inicio Rápido

La forma más rápida de comenzar es con la plantilla oficial [rayforge-addon-template](https://github.com/barebaric/rayforge-addon-template). Haz un fork o clónala, renombra el directorio y actualiza los metadatos para que coincidan con tu addon.

## Cómo Funcionan los Addons

El `AddonManager` escanea el directorio `addons` en busca de addons válidos. Un addon es simplemente un directorio que contiene un archivo manifiesto `rayforge-addon.yaml` junto con tu código Python.

Así es como se ve un addon típico:

```text
my-rayforge-addon/
├── rayforge-addon.yaml  <-- Manifiesto requerido
├── my_addon/            <-- Tu paquete Python
│   ├── __init__.py
│   ├── backend.py       <-- Punto de entrada del backend
│   └── frontend.py      <-- Punto de entrada del frontend (opcional)
├── assets/              <-- Recursos opcionales
├── locales/             <-- Traducciones opcionales (archivos .po)
└── README.md
```

## Tu Primer Addon

Vamos a crear un addon simple que registre un controlador de máquina personalizado. Primero, crea el manifiesto:

```yaml title="rayforge-addon.yaml"
name: my_laser_driver
display_name: "My Laser Driver"
description: "Adds support for the XYZ laser cutter."
api_version: 9

author:
  name: Jane Doe
  email: jane@example.com

provides:
  backend: my_addon.backend
```

Ahora crea el módulo del backend que registra tu controlador:

```python title="my_addon/backend.py"
import pluggy

hookimpl = pluggy.HookimplMarker("rayforge")

@hookimpl
def register_machines(machine_manager):
    """Register our custom machine driver."""
    from .my_driver import MyLaserMachine
    machine_manager.register("my_laser", MyLaserMachine)
```

¡Eso es todo! Tu addon se cargará cuando Rayforge se inicie, y tu controlador de máquina estará disponible para los usuarios.

La documentación del [Manifiesto](./addon-manifest.md) cubre todas las opciones de configuración disponibles.

## Entendiendo los Puntos de Entrada

Los addons pueden proporcionar dos puntos de entrada, cada uno cargado en diferentes momentos:

El punto de entrada del **backend** se carga tanto en el proceso principal como en los procesos de trabajo. Úsalo para controladores de máquinas, tipos de pasos, productores y transformadores de ops, o cualquier funcionalidad principal que no necesite dependencias de interfaz de usuario.

El punto de entrada del **frontend** solo se carga en el proceso principal. Aquí es donde pondrías componentes de interfaz de usuario, widgets GTK, elementos de menú y cualquier cosa que necesite acceso a la ventana principal.

Ambos se especifican como rutas de módulo con puntos como `my_addon.backend`.

## Conectando con Rayforge mediante Hooks

Rayforge usa hooks de `pluggy` para permitir que los addons se integren con la aplicación. Simplemente decora tus funciones con `@pluggy.HookimplMarker("rayforge")`:

```python
import pluggy
from rayforge.context import RayforgeContext

hookimpl = pluggy.HookimplMarker("rayforge")

@hookimpl
def rayforge_init(context: RayforgeContext):
    """Called when Rayforge is fully initialized."""
    # Your setup code here
    pass

@hookimpl
def on_unload():
    """Called when the addon is being disabled or unloaded."""
    # Clean up resources here
    pass
```

La documentación de [Hooks](./addon-hooks.md) describe cada hook disponible y cuándo se llama.

## Registrando tus Componentes

La mayoría de los hooks reciben un objeto registro que usas para registrar tus componentes personalizados:

```python
@hookimpl
def register_steps(step_registry):
    from .my_step import MyCustomStep
    step_registry.register(MyCustomStep)

@hookimpl
def register_actions(action_registry):
    from .actions import setup_actions
    setup_actions(action_registry)
```

La documentación de [Registros](./addon-registries.md) explica cada registro y cómo usarlos.

## Accediendo a los Datos de Rayforge

El hook `rayforge_init` te da acceso a un objeto `RayforgeContext`. A través de este contexto, puedes acceder a todo en Rayforge:

Puedes obtener la máquina actualmente activa mediante `context.machine`, o acceder a todas las máquinas a través de `context.machine_mgr`. El objeto `context.config` contiene la configuración global, mientras que `context.camera_mgr` proporciona acceso a las fuentes de cámara. Para materiales, usa `context.material_mgr`, y para recetas de procesamiento, usa `context.recipe_mgr`. El gestor de dialectos G-code está disponible como `context.dialect_mgr`, y las funciones de IA van a través de `context.ai_provider_mgr`. Para localización, consulta `context.language` para el código de idioma actual. El gestor de addons en sí está disponible como `context.addon_mgr`, y si estás creando addons de pago, `context.license_validator` maneja la validación de licencias.

## Agregando Traducciones

Los addons pueden proporcionar traducciones usando archivos `.po` estándar. Organízalos así:

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

Rayforge compila automáticamente los archivos `.po` a archivos `.mo` cuando tu addon se carga.

## Probando Durante el Desarrollo

Para probar tu addon localmente, crea un enlace simbólico desde tu carpeta de desarrollo al directorio de addons de Rayforge.

Primero, encuentra tu directorio de configuración. En Windows, es `C:\Users\<User>\AppData\Local\rayforge\rayforge\addons`. En macOS, busca en `~/Library/Application Support/rayforge/addons`. En Linux, es `~/.config/rayforge/addons`.

Luego crea el enlace simbólico:

```bash
ln -s /path/to/my-rayforge-addon ~/.config/rayforge/addons/my-rayforge-addon
```

Reinicia Rayforge y revisa la consola por un mensaje como `Loaded addon: my_laser_driver`.

## Compartiendo tu Addon

Cuando estés listo para compartir tu addon, súbelo a un repositorio Git público en GitHub o GitLab. Luego envíalo al [rayforge-registry](https://github.com/barebaric/rayforge-registry) haciendo un fork del repositorio, agregando los metadatos de tu addon y abriendo un pull request.

Una vez aceptado, los usuarios podrán instalar tu addon directamente a través del gestor de addons de Rayforge.
