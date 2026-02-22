# Obteniendo el Código

Esta guía cubre cómo obtener el código fuente de Rayforge para desarrollo.

## Hacer Fork del Repositorio

Haz fork del [repositorio Rayforge](https://github.com/barebaric/rayforge) en GitHub para crear tu propia copia donde puedas hacer cambios.

## Clonar Tu Fork

```bash
git clone https://github.com/YOUR_USERNAME/rayforge.git
cd rayforge
```

## Añadir Repositorio Upstream

Añade el repositorio original como un remote upstream para mantener seguimiento de cambios:

```bash
git remote add upstream https://github.com/barebaric/rayforge.git
```

## Verificar el Repositorio

Verifica que los remotes estén configurados correctamente:

```bash
git remote -v
```

Deberías ver tanto tu fork (origin) como el repositorio upstream.

## Siguientes Pasos

Después de obtener el código, continúa con [Configuración](setup) para configurar tu entorno de desarrollo.
