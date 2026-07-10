# Configuración

Esta guía cubre la configuración de tu entorno de desarrollo para Rayforge.

## Linux

### Prerrequisitos

Ver la [Guía de Instalación](../../getting-started/installation.mdx#linux-pixi) para instrucciones de instalación de Pixi.

### Hooks Pre-commit (Opcional)

Para formatear y analizar tu código automáticamente antes de cada commit, puedes instalar hooks pre-commit:

```bash
pixi run pre-commit-install
```

### Comandos Útiles

Todos los comandos se ejecutan via `pixi run`:

-   `pixi run rayforge`: Ejecutar la aplicación.
    -   Añade `--loglevel=DEBUG` para salida más verbosa.
-   `pixi run test`: Ejecutar la suite de pruebas completa con `pytest`.
-   `pixi run format`: Formatear todo el código usando `ruff`.
-   `pixi run lint`: Ejecutar todos los linters (`flake8`, `pyflakes`, `pyright`).

## Windows

### Prerrequisitos

Ver la [Guía de Instalación](../../getting-started/installation.mdx#windows-developer) para instrucciones detalladas de configuración de desarrollo MSYS2.

### Inicio Rápido

Las tareas de desarrollo en Windows se gestionan via el script `run.bat`, que es un wrapper para el shell de MSYS2.

Después de clonar el repositorio y completar la configuración de MSYS2, puedes usar estos comandos desde un Command Prompt estándar de Windows o PowerShell:

```batch
.\run.bat setup
```

Esto ejecuta `scripts/win/win_setup.sh` para instalar todos los paquetes necesarios del sistema y Python en tu entorno MSYS2/MinGW64.

### Hooks Pre-commit (Opcional)

Para formatear y analizar automáticamente tu código antes de cada commit, ejecuta esto desde el shell MSYS2 MINGW64:

```bash
bash scripts/win/win_setup_dev.sh
```

:::note

Los hooks pre-commit requieren ejecutar comandos git dentro del shell MSYS2 MINGW64, no desde PowerShell o Símbolo del sistema.

:::

### Comandos Útiles

Todos los comandos se ejecutan via el script `run.bat`:

-   `run app`: Ejecutar la aplicación desde el código fuente.
    -   Añade `--loglevel=DEBUG` para salida más verbosa.
-   `run test`: Ejecutar la suite de pruebas completa usando `pytest`.
-   `run lint`: Ejecutar todos los linters (`flake8`, `pyflakes`, `pyright`).
-   `run format`: Formatear y corregir código automáticamente usando `ruff`.
-   `run build`: Construir el ejecutable final de Windows (`.exe`).

Alternativamente, puedes ejecutar los scripts directamente desde el shell MSYS2 MINGW64:

-   `bash scripts/win/win_run.sh`: Ejecutar la aplicación.
-   `bash scripts/win/win_test.sh`: Ejecutar la suite de pruebas.
-   `bash scripts/win/win_lint.sh`: Ejecutar todos los linters.
-   `bash scripts/win/win_format.sh`: Formatear y corregir código automáticamente.
-   `bash scripts/win/win_build.sh`: Construir el ejecutable de Windows.
