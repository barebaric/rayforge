---
slug: 5-tips-better-engraving
title: 5 Consejos para Mejores Resultados de Grabado Láser con Rayforge
authors: rayforge_team
tags: [engraving, optimization, quality, workflow]
---

![3D Preview](/screenshots/main-3d.png)

Obtener resultados de grabado láser de calidad profesional requiere más que solo buen hardware—tu configuración de software y flujo de trabajo también importan. Aquí hay cinco consejos para ayudarte a aprovechar al máximo Rayforge.

<!-- truncate -->

## 1. Usa Overscan para un Grabado Raster Más Suave

Al hacer grabado raster, un problema común es ver líneas visibles o inconsistencias en los bordes donde el láser cambia de dirección. Esto sucede porque la cabeza del láser necesita desacelerar y acelerar, lo que puede afectar la calidad del grabado.

**Solución**: Activa **Overscan** en la configuración de tu operación raster.

Overscan extiende la trayectoria de viaje del láser más allá del área de grabado real, permitiendo que la cabeza alcance la velocidad máxima antes de entrar al área de trabajo y mantenga esa velocidad durante todo el recorrido. Esto resulta en un grabado mucho más suave y consistente.

Para activar overscan:

1. Selecciona tu operación raster
2. Abre la configuración de la operación
3. Activa "Overscan" y establece la distancia (típicamente 3-5mm funciona bien)

Aprende más en nuestra [guía de Overscan](/docs/features/overscan).

## 2. Optimiza el Tiempo de Desplazamiento con Ordenamiento de Trayectorias

Para operaciones de contorno con muchas trayectorias separadas, el orden en que el láser visita cada forma puede impactar significativamente el tiempo total del trabajo.

**Solución**: Usa la **optimización del tiempo de desplazamiento** integrada de Rayforge.

Rayforge puede reordenar automáticamente las trayectorias para minimizar el tiempo de desplazamiento sin corte. Esto es especialmente útil para trabajos con muchos objetos pequeños o texto con múltiples letras.

La optimización de trayectorias típicamente está activada por defecto, pero puedes verificarla y ajustarla en la configuración de la operación Contour.

## 3. Agrega Pestañas de Sujeción para Prevenir el Movimiento de Piezas

Nada es más frustrante que tener un trabajo de corte casi terminado arruinado porque la pieza se movió o cayó a través de la cama de la máquina en el último momento.

**Solución**: Usa **Pestañas de Sujeción** para mantener las piezas en su lugar hasta que el trabajo se complete.

Las pestañas de sujeción son pequeñas secciones sin cortar que mantienen tu pieza conectada al material circundante. Después de que el trabajo se completa, puedes remover fácilmente la pieza y limpiar las pestañas con un cuchillo o papel de lija.

Rayforge soporta tanto la colocación manual como automática de pestañas:

- **Manual**: Haz clic exactamente donde quieres las pestañas en el lienzo
- **Automática**: Especifica el número de pestañas y deja que Rayforge las distribuya uniformemente

Consulta la [documentación de Pestañas de Sujeción](/docs/features/holding-tabs) para una guía completa.

## 4. Previsualiza tu Trabajo en 3D Antes de Ejecutarlo

Una de las funciones más valiosas de Rayforge es la vista previa 3D de G-code. Es tentador saltarse este paso y enviar el trabajo directamente a la máquina, pero tomar un momento para previsualizar puede ahorrarte tiempo y materiales.

**Qué buscar en la vista previa**:

- Verificar que todas las operaciones se ejecuten en el orden correcto
- Revisar si hay trayectorias inesperadas o superposiciones
- Confirmar que las operaciones de múltiples pasadas tengan el número correcto de pasadas
- Asegurar que los límites del trabajo quepan dentro de tu material

Para abrir la vista previa 3D, haz clic en el botón **3D Preview** en la barra de herramientas principal después de generar tu G-code.

Aprende más sobre la vista previa 3D en nuestra [documentación de UI](/docs/ui/3d-preview).

## 5. Usa Ganchos de G-code Personalizados para Flujos de Trabajo Consistentes

Si te encuentras ejecutando los mismos comandos G-code antes o después de cada trabajo—como hacer homing, activar una asistencia de aire, o ejecutar una rutina de enfoque—puedes automatizar esto con **Macros y Ganchos de G-code**.

**Casos de uso comunes**:

- **Gancho pre-trabajo**: Hacer homing de la máquina, activar asistencia de aire, ejecutar una rutina de auto-enfoque
- **Gancho post-trabajo**: Desactivar asistencia de aire, volver a la posición home, reproducir un sonido de finalización
- **Macros específicas por capa**: Cambiar la altura de enfoque entre operaciones, cambiar módulos láser

Los ganchos soportan sustitución de variables, por lo que puedes referenciar propiedades del trabajo como grosor del material, tipo de operación, y más.

Ejemplo de gancho pre-trabajo:

```gcode
G28 ; Home all axes
M8 ; Turn on air assist
G0 Z{focus_height} ; Move to focus height
```

Consulta nuestra [guía de Macros y Ganchos de G-code](/docs/machine/hooks-macros) para ejemplos detallados y referencia de variables.

---

## Consejo Extra: Prueba Primero en Material de Desecho

Aunque esto no es específico de Rayforge, vale la pena repetirlo: siempre prueba nuevas configuraciones, operaciones o materiales primero en desechos. Usa los perfiles de materiales y preajustes de operación de Rayforge para guardar tus configuraciones probadas para uso futuro.

---

*¿Tienes tus propios consejos y trucos de Rayforge? ¡Compártelos con la comunidad en [GitHub Discussions](https://github.com/barebaric/rayforge/discussions)!*
