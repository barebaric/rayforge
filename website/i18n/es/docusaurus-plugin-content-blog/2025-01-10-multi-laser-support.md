---
slug: multi-laser-support
title: Soporte Multi-Láser - Elige Diferentes Láseres para Cada Operación
authors: rayforge_team
tags: [multi-laser, operations, workflow]
---

![Camera Overlay](/images/camera-overlay.png)

Una de las funciones más potentes de Rayforge es la capacidad de asignar diferentes láseres a diferentes operaciones dentro de un solo trabajo. Esto abre posibilidades emocionantes para configuraciones de múltiples herramientas y flujos de trabajo especializados.

<!-- truncate -->

## ¿Qué es el Soporte Multi-Láser?

Si tu máquina está equipada con múltiples módulos láser—por ejemplo, un láser de diodo para grabar y un láser CO2 para cortar, o diferentes láseres de diodo de potencia optimizados para diferentes materiales—Rayforge te permite aprovechar al máximo esta configuración.

Con el soporte multi-láser, puedes:

- **Asignar diferentes láseres a diferentes operaciones** en tu trabajo
- **Cambiar entre módulos láser** automáticamente durante la ejecución del trabajo
- **Optimizar para material y tarea** usando la herramienta correcta para cada operación

## Casos de Uso

### Grabado y Corte Híbrido

Imagina que estás trabajando en un proyecto de letrero de madera:

1. **Operación 1**: Usar un láser de diodo de baja potencia para grabar texto fino y gráficos detallados
2. **Operación 2**: Cambiar a un láser CO2 de mayor potencia para cortar la forma del letrero

Con Rayforge, simplemente asignas cada operación al láser apropiado en tu perfil de máquina, y el software se encarga del resto.

### Optimización Específica por Material

Diferentes tipos de láser sobresalen en diferentes materiales:

- **Láseres de diodo**: Geniales para grabado en madera, cuero y algunos plásticos
- **Láseres CO2**: Excelentes para cortar acrílico, madera y otros materiales orgánicos
- **Láseres de fibra**: Perfectos para marcar metales

Si tienes múltiples tipos de láser en un sistema pórtico, el soporte multi-láser de Rayforge te permite usar la herramienta óptima para cada parte de tu proyecto.

## Cómo Configurarlo

### 1. Configura Múltiples Láseres en tu Perfil de Máquina

Ve a **Machine Setup → Multiple Lasers** y define cada módulo láser en tu sistema. Puedes especificar:

- Tipo de láser y rango de potencia
- Posiciones de desplazamiento (si los láseres están montados en diferentes ubicaciones)
- Compatibilidad de materiales

Consulta nuestra [Guía de Configuración de Láser](/docs/machine/laser) para instrucciones detalladas.

### 2. Asigna Láseres a Operaciones

Al crear operaciones en tu proyecto:

1. Selecciona la operación (Contorno, Raster, etc.)
2. En la configuración de la operación, elige qué láser usar desde el menú desplegable
3. Configura los parámetros de la operación específicos para ese láser

### 3. Vista Previa y Ejecución

Usa la vista previa 3D para verificar tus trayectorias, luego envía el trabajo a tu máquina. Rayforge generará automáticamente los comandos G-code apropiados para cambiar entre láseres según sea necesario.

## Detalles Técnicos

Bajo el capó, Rayforge usa comandos G-code para cambiar entre módulos láser. La implementación exacta depende de tu firmware y configuración de hardware, pero los enfoques comunes incluyen:

- **M3/M4 con desplazamientos de herramienta**: Cambiar entre láseres usando comandos de cambio de herramienta
- **Control GPIO**: Usar pines GPIO soportados por el firmware para habilitar/deshabilitar diferentes módulos láser
- **Macros personalizados**: Definir macros pre y post-operación para cambio de láser

## Primeros Pasos

El soporte multi-láser está disponible en Rayforge 0.15 y versiones posteriores. Para comenzar:

1. Actualiza a la última versión
2. Configura tu perfil de máquina con múltiples láseres
3. ¡Pruébalo en un proyecto de prueba!

Consulta la [documentación de Perfiles de Máquina](/docs/machine/general) para más detalles.

---

*¿Tienes una configuración multi-láser? ¡Nos encantaría saber sobre tu experiencia! Comparte tus proyectos y comentarios en [GitHub](https://github.com/barebaric/rayforge).*
