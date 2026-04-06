# Proveedor de IA

![Configuración de Proveedor de IA](/screenshots/application-ai.png)

Configura proveedores de IA que los addons pueden usar para añadir
funciones inteligentes a Rayforge.

## Cómo Funciona

Los addons pueden usar los proveedores de IA configurados sin necesidad
de sus propias claves API. Esto centraliza tu configuración de IA y te
permite controlar qué proveedores están disponibles para los addons.

## Añadir un Proveedor

1. Haz clic en **Añadir Proveedor** para crear una nueva configuración
2. Introduce un **Nombre** para identificar este proveedor
3. Establece la **URL Base** al endpoint API de tu servicio de IA
4. Introduce tu **Clave API** para la autenticación
5. Especifica un **Modelo por Defecto** para usar con este proveedor
6. Haz clic en **Probar** para verificar que tu configuración funciona

## Tipos de Proveedor

### Compatible con OpenAI

Este tipo de proveedor funciona con cualquier servicio que use el formato
API de OpenAI. Esto incluye varios proveedores en la nube y soluciones
autohospedadas.

La URL base por defecto está configurada para la API de OpenAI, pero
puedes cambiarla a cualquier servicio compatible.

## Gestionar Proveedores

- **Activar/Desactivar**: Activa o desactiva un proveedor sin eliminarlo
- **Establecer por Defecto**: Haz clic en el icono de verificación para
  hacer que un proveedor sea el predeterminado
- **Eliminar**: Elimina un proveedor que ya no necesitas

:::warning
Tus claves API se almacenan localmente en tu ordenador y nunca se
comparten con terceros.
:::

## Temas Relacionados

- [Addons](addons) - Instalar y gestionar addons
- [Máquinas](machines) - Configuración de máquinas
- [Materiales](materials) - Bibliotecas de materiales
