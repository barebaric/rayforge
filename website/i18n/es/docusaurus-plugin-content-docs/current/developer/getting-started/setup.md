# Configuración

Esta guía cubre la configuración de tu entorno de desarrollo para Rayforge.

## Linux

### Prerrequisitos

Ver la [Guía de Instalación](../../getting-started/installation#linux-pixi) para instrucciones de instalación de Pixi.

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

-   [MSYS2](https://www.msys2.org/) (proporciona el entorno MinGW64).
-   [Git for Windows](https://git-scm.com/download/win).

### Instalar

Las tareas de desarrollo en Windows se gestionan via el script `run.bat`, que es un wrapper para el shell de MSYS2.

Después de clonar el repositorio, ejecuta el comando de configuración desde un Command Prompt estándar de Windows o PowerShell:

```batch
.\run.bat setup
```

Esto ejecuta `scripts/win/win_setup.sh` para instalar todos los paquetes necesarios del sistema y Python en tu entorno MSYS2/MinGW64.

### Comandos Útiles

Todos los comandos se ejecutan via el script `run.bat`:

-   `run app`: Ejecutar la aplicación desde el código fuente.
    -   Añade `--loglevel=DEBUG` para salida más verbosa.
-   `run test`: Ejecutar la suite de pruebas completa usando `pytest`.
-   `run build`: Construir el ejecutable final de Windows (`.exe`).
