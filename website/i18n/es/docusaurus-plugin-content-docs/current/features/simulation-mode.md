# Modo Simulación

![Modo Simulación](/screenshots/main-simulation.png)

El Modo Simulación proporciona visualización en tiempo real de la ejecución de tu trabajo láser antes de ejecutarlo en la máquina real. Muestra orden de ejecución, variaciones de velocidad y niveles de potencia a través de una superposición interactiva en la vista 2D.

## Resumen

El Modo Simulación te ayuda a:

- **Visualizar orden de ejecución** - Ver la secuencia exacta en que se ejecutarán las operaciones
- **Identificar variaciones de velocidad** - Mapa de calor de colores muestra movimientos lentos (azul) a rápidos (rojo)
- **Verificar niveles de potencia** - La transparencia indica potencia (tenue=baja, fuerte=alta)
- **Validar pruebas de material** - Confirmar orden de ejecución de cuadrícula de prueba
- **Detectar errores temprano** - Encontrar problemas antes de desperdiciar material
- **Entender tiempos** - Ver cuánto tiempo toman diferentes operaciones


## Activar Modo Simulación

Hay tres formas de entrar al Modo Simulación:

### Método 1: Atajo de Teclado
Presiona <kbd>f7</kbd> para alternar modo simulación on/off.

### Método 2: Menú
- Navega a **Ver → Simular Ejecución**
- Haz clic para alternar on/off

### Método 3: Barra de Herramientas (si disponible)
- Haz clic en el botón de modo simulación en la barra de herramientas

:::note Solo Vista 2D
El modo simulación funciona en vista 2D. Si estás en vista 3D (<kbd>f6</kbd>), cambia a vista 2D (<kbd>f5</kbd>) primero.
:::


## Entendiendo la Visualización

### Mapa de Calor de Velocidad

Las operaciones se colorean basándose en su velocidad:

| Color  | Velocidad | Significado                          |
| ------ | --------- | ------------------------------------ |
| 🔵 **Azul** | Más lenta | Velocidad mínima en tu trabajo       |
| 🔵 **Cian** | Lenta     | Por debajo del promedio de velocidad |
| 🟢 **Verde** | Media    | Velocidad promedio                   |
| 🟡 **Amarillo** | Rápida | Por encima del promedio de velocidad |
| 🔴 **Rojo** | Más rápida | Velocidad máxima en tu trabajo      |

El mapa de calor está **normalizado** al rango de velocidad real de tu trabajo:
- Si tu trabajo corre a 100-1000 mm/min, azul=100, rojo=1000
- Si tu trabajo corre a 5000-10000 mm/min, azul=5000, rojo=10000


### Transparencia de Potencia

La opacidad de línea indica potencia del láser:

- **Líneas tenues** (10% opacidad) = Baja potencia (0%)
- **Translúcido** (50% opacidad) = Potencia media (50%)
- **Líneas sólidas** (100% opacidad) = Potencia completa (100%)

Esto ayuda a identificar:
- Movimientos de viaje (0% potencia) - Muy tenues
- Operaciones de grabado - Opacidad moderada
- Operaciones de corte - Líneas sólidas y fuertes

### Indicador de Cabezal Láser

La posición del láser se muestra con una cruz:

- 🔴 Cruz roja (líneas de 6mm)
- Contorno de círculo (radio de 3mm)
- Punto central (0.5mm)

El indicador se mueve durante la reproducción, mostrando exactamente dónde está el láser en la secuencia de ejecución.

## Controles de Reproducción

Cuando el modo simulación está activo, aparecen controles de reproducción en la parte inferior del lienzo:


### Botón Reproducir/Pausar

- **▶️ Reproducir**: Inicia reproducción automática
- **⏸️ Pausar**: Se detiene en la posición actual
- **Auto-reproducción**: La reproducción inicia automáticamente cuando habilitas modo simulación

### Deslizador de Progreso

- **Arrastra** para desplazarte por la ejecución
- **Haz clic** para saltar a un punto específico
- Muestra paso actual / pasos totales
- Soporta posiciones fraccionales para desplazamiento suave

### Pantalla de Rango de Velocidad

Muestra las velocidades mínima y máxima en tu trabajo:

```
Rango de velocidad: 100 - 5000 mm/min
```

Esto te ayuda a entender los colores del mapa de calor.

## Usando Modo Simulación

### Validar Orden de Ejecución

La simulación muestra el orden exacto en que se ejecutarán las operaciones:

1. Habilita modo simulación (<kbd>f7</kbd>)
2. Observa la reproducción
3. Verifica que las operaciones se ejecuten en la secuencia esperada
4. Confirma que los cortes ocurran después del grabado (si aplica)

**Ejemplo:** Cuadrícula de prueba de material
- Observa el orden optimizado por riesgo (velocidades más rápidas primero)
- Confirma que celdas de baja potencia se ejecuten antes que las de alta potencia
- Valida que la prueba corra en secuencia segura

### Verificar Variaciones de Velocidad

Usa el mapa de calor para identificar cambios de velocidad:

- **Color consistente** = Velocidad uniforme (bueno para grabado)
- **Cambios de color** = Variaciones de velocidad (esperado en esquinas)
- **Áreas azules** = Movimientos lentos (verifica si es intencional)

### Estimar Tiempo de Trabajo

La duración de reproducción está escalada a 5 segundos para el trabajo completo:

- Observa la velocidad de reproducción
- Estima tiempo real: Si la reproducción se siente fluida, el trabajo será rápido
- Si la reproducción salta rápidamente, el trabajo tiene muchos segmentos pequeños

:::tip Tiempo Real
 Para tiempo real del trabajo durante ejecución (no simulación), revisa la sección
 derecha de la barra de estado después de generar G-code.
 :::


### Depurar Pruebas de Material

Para cuadrículas de prueba de material, la simulación muestra:

1. **Orden de ejecución** - Verifica que las celdas corran de más rápida→más lenta
2. **Mapa de calor de velocidad** - Cada columna debería ser un color diferente
3. **Transparencia de potencia** - Cada fila debería tener opacidad diferente

Esto ayuda a confirmar que la prueba correrá correctamente antes de usar material.

## Editar Mientras Simulas

A diferencia de muchas herramientas CAM, Rayforge te permite **editar piezas de trabajo durante la simulación**:

- Mover, escalar, rotar objetos ✅
- Cambiar ajustes de operación ✅
- Añadir/remover piezas de trabajo ✅
- Acercar y desplazar ✅

**Actualización automática:** La simulación se refresca automáticamente cuando cambias ajustes.

:::note Sin Cambio de Contexto
Puedes permanecer en modo simulación mientras editas - no necesitas alternar de un lado a otro.
:::


## Consejos y Mejores Prácticas

### Cuándo Usar Simulación

✅ **Siempre simula antes de:**
- Ejecutar materiales costosos
- Trabajos largos (>30 minutos)
- Cuadrículas de prueba de material
- Trabajos con orden de ejecución complejo

✅ **Usa simulación para:**
- Verificar orden de operación
- Buscar movimientos de viaje inesperados
- Validar ajustes de velocidad/potencia
- Entrenar nuevos usuarios

### Leer la Visualización

✅ **Busca:**
- Colores consistentes dentro de operaciones (bueno)
- Transiciones suaves entre segmentos (bueno)
- Áreas azules inesperadas (investiga - ¿por qué tan lento?)
- Líneas tenues en áreas de corte (mal - verifica ajustes de potencia)

⚠️ **Banderas rojas:**
- Cortar antes de grabar (la pieza de trabajo puede moverse)
- Secciones azules (lentas) muy largas (ineficiente)
- Cambios de potencia a mitad de operación (verifica ajustes)

### Consejos de Rendimiento

- La simulación se actualiza automáticamente con cambios
- Para trabajos muy complejos (1000+ operaciones), la simulación puede volverse lenta
- Deshabilita simulación (<kbd>f7</kbd>) cuando no se necesite para mejor rendimiento

## Atajos de Teclado

| Atajo     | Acción                                    |
| --------- | ----------------------------------------- |
| <kbd>f7</kbd> | Alternar modo simulación on/off       |
| <kbd>f5</kbd> | Cambiar a vista 2D (requerido para simulación) |
| <kbd>espacio</kbd> | Reproducir/Pausar reproducción       |
| <kbd>izquierda</kbd> | Paso hacia atrás                    |
| <kbd>derecha</kbd> | Paso hacia adelante                  |
| <kbd>inicio</kbd> | Saltar al inicio                      |
| <kbd>fin</kbd> | Saltar al final                         |

## Temas Relacionados

- **[Previsualización 3D](../ui/3d-preview)** - Visualización 3D de trayectoria de herramienta
- **[Cuadrícula de Prueba de Material](operations/material-test-grid)** - Usar simulación para validar pruebas
- **[Simulando Tu Trabajo](simulating-your-job)** - Guía detallada de simulación
