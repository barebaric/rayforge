# Manifiesto del Addon

Cada addon necesita un archivo `rayforge-addon.yaml` en su directorio raíz. Este manifiesto le informa a Rayforge sobre tu addon—su nombre, qué proporciona y cómo cargarlo.

## Estructura Básica

Aquí hay un manifiesto completo con todos los campos comunes:

```yaml
name: my_custom_addon
display_name: "My Custom Addon"
description: "Adds support for the XYZ laser cutter."
api_version: 9
url: https://github.com/username/my-custom-addon

author:
  name: Jane Doe
  email: jane@example.com

depends:
  - rayforge>=0.27.0

requires:
  - some-other-addon>=1.0.0

provides:
  backend: my_addon.backend
  frontend: my_addon.frontend
  assets:
    - path: assets/profiles.json
      type: profiles

license:
  name: MIT
```

## Campos Requeridos

### `name`

Un identificador único para tu addon. Debe ser un nombre de módulo Python válido—solo letras, números y guiones bajos, y no puede comenzar con un número.

```yaml
name: my_custom_addon
```

### `display_name`

Un nombre legible para humanos que se muestra en la interfaz de usuario. Puede contener espacios y caracteres especiales.

```yaml
display_name: "My Custom Addon"
```

### `description`

Una breve descripción de lo que hace tu addon. Aparece en el gestor de addons.

```yaml
description: "Adds support for the XYZ laser cutter."
```

### `api_version`

La versión de API que tu addon utiliza. Debe ser al menos 1 (la versión mínima soportada) y como máximo la versión actual (9). Usar una versión más alta de la soportada hará que tu addon falle la validación.

```yaml
api_version: 9
```

Consulta la documentación de [Hooks](./addon-hooks.md#api-version-history) para ver qué cambió en cada versión.

### `author`

Información sobre el autor del addon. El campo `name` es requerido; `email` es opcional pero recomendado para que los usuarios puedan contactarte.

```yaml
author:
  name: Jane Doe
  email: jane@example.com
```

## Campos Opcionales

### `url`

Una URL a la página principal o repositorio de tu addon.

```yaml
url: https://github.com/username/my-custom-addon
```

### `depends`

Restricciones de versión para Rayforge mismo. Especifica la versión mínima que tu addon requiere.

```yaml
depends:
  - rayforge>=0.27.0
```

### `requires`

Dependencias en otros addons. Lista nombres de addons con restricciones de versión.

```yaml
requires:
  - some-other-addon>=1.0.0
```

### `version`

El número de versión de tu addon. Típicamente se determina automáticamente desde las etiquetas de git, pero puedes especificarlo explícitamente. Usa versionamiento semántico (ej., `1.0.0`).

```yaml
version: 1.0.0
```

## Puntos de Entrada

La sección `provides` define lo que tu addon contribuye a Rayforge.

### Backend

El módulo del backend se carga tanto en el proceso principal como en los procesos de trabajo. Úsalo para controladores de máquinas, tipos de pasos, productores de ops y cualquier funcionalidad principal.

```yaml
provides:
  backend: my_addon.backend
```

El valor es una ruta de módulo Python con puntos relativa al directorio de tu addon.

### Frontend

El módulo del frontend solo se carga en el proceso principal. Úsalo para componentes de interfaz de usuario, widgets GTK y cualquier cosa que necesite la ventana principal.

```yaml
provides:
  frontend: my_addon.frontend
```

### Assets

Puedes empaquetar archivos de assets que Rayforge reconocerá. Cada asset tiene una ruta y un tipo:

```yaml
provides:
  assets:
    - path: assets/profiles.json
      type: profiles
    - path: assets/templates
      type: templates
```

La `path` es relativa a la raíz de tu addon y debe existir. Los tipos de assets son definidos por Rayforge y pueden incluir cosas como perfiles de máquinas, bibliotecas de materiales o plantillas.

## Información de Licencia

El campo `license` describe cómo está licenciado tu addon. Para addons gratuitos, simplemente especifica el nombre de la licencia usando un identificador SPDX:

```yaml
license:
  name: MIT
```

Los identificadores SPDX comunes incluyen `MIT`, `Apache-2.0`, `GPL-3.0` y `BSD-3-Clause`.

## Addons de Pago

Rayforge soporta addons de pago a través de la validación de licencias de Gumroad. Si quieres vender tu addon, puedes configurarlo para requerir una licencia válida antes de que funcione.

### Configuración Básica de Pago

```yaml
license:
  name: BSL-1.1
  required: true
  purchase_url: https://gum.co/my-addon
```

Cuando `required` es verdadero, Rayforge verificará una licencia válida antes de cargar tu addon. La `purchase_url` se muestra a los usuarios que no tienen una licencia.

### ID de Producto de Gumroad

Agrega tu ID de producto de Gumroad para habilitar la validación de licencias:

```yaml
license:
  name: BSL-1.1
  required: true
  purchase_url: https://gum.co/my-addon
  product_id: "abc123def456"
```

Para múltiples IDs de producto (ej., diferentes niveles de precios):

```yaml
license:
  name: BSL-1.1
  required: true
  purchase_url: https://gum.co/my-addon
  product_ids:
    - "abc123def456"
    - "xyz789ghi012"
```

### Ejemplo Completo de Addon de Pago

Aquí hay un manifiesto completo para un addon de pago:

```yaml
name: premium_laser_pack
display_name: "Premium Laser Pack"
description: "Advanced features for professional laser cutting."
api_version: 9
url: https://example.com/premium-laser-pack

author:
  name: Your Name
  email: you@example.com

depends:
  - rayforge>=0.27.0

provides:
  backend: premium_pack.backend
  frontend: premium_pack.frontend

license:
  name: BSL-1.1
  required: true
  purchase_url: https://gum.co/premium-laser-pack
  product_ids:
    - "standard_tier_id"
    - "pro_tier_id"
```

### Verificando el Estado de la Licencia en Código

En el código de tu addon, puedes verificar si una licencia es válida:

```python
@hookimpl
def rayforge_init(context):
    if context.license_validator:
        # Check if user has a valid license for your product
        is_valid = context.license_validator.is_product_valid("your_product_id")
        if not is_valid:
            # Optionally show a message or limit functionality
            logger.warning("License not found - some features disabled")
```

## Reglas de Validación

Rayforge valida tu manifiesto al cargar el addon. Aquí están las reglas:

El `name` debe ser un identificador Python válido (letras, números, guiones bajos, sin números al inicio). El `api_version` debe ser un entero entre 1 y la versión actual. El `author.name` no puede estar vacío ni contener texto de marcador como "your-github-username". Los puntos de entrada deben ser rutas de módulo válidas y los módulos deben existir. Las rutas de assets deben ser relativas (sin `..` o `/` al inicio) y los archivos deben existir.

Si la validación falla, Rayforge registra un error y omite tu addon. Revisa la salida de la consola durante el desarrollo para detectar estos problemas.
