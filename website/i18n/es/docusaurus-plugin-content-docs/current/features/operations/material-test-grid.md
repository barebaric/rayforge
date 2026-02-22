# Cuadrícula de Prueba de Material

El generador de Cuadrícula de Prueba de Material crea patrones de prueba paramétricos para ayudarte a encontrar los ajustes óptimos de láser para diferentes materiales.

## Resumen

Las pruebas de material son esenciales para el trabajo láser - diferentes materiales requieren diferentes ajustes de potencia y velocidad. La Cuadrícula de Prueba de Material automatiza este proceso:

- Generando cuadrículas de prueba con rangos configurables de velocidad/potencia
- Proporcionando preajustes para tipos comunes de láser (Diodo, CO2)
- Optimizando el orden de ejecución para seguridad (velocidades más rápidas primero)
- Añadiendo etiquetas para identificar los ajustes de cada celda de prueba

## Crear una Cuadrícula de Prueba de Material

### Paso 1: Abrir el Generador

Accede al generador de Cuadrícula de Prueba de Material:

- Menú: **Herramientas → Cuadrícula de Prueba de Material**
- Esto crea una pieza de trabajo especial que genera el patrón de prueba

### Paso 2: Elegir un Preajuste (Opcional)

Rayforge incluye preajustes para escenarios comunes:

| Preajuste          | Rango de Velocidad    | Rango de Potencia | Usar Para               |
| ------------------ | --------------------- | ----------------- | ----------------------- |
| **Grabado Diodo**  | 1000-10000 mm/min     | 10-100%           | Grabado láser diodo     |
| **Corte Diodo**    | 100-5000 mm/min       | 50-100%           | Corte láser diodo       |
| **Grabado CO2**    | 3000-20000 mm/min     | 10-50%            | Grabado láser CO2       |
| **Corte CO2**      | 1000-20000 mm/min     | 30-100%           | Corte láser CO2         |

Los preajustes son puntos de partida - puedes ajustar todos los parámetros después de seleccionar uno.

### Paso 3: Configurar Parámetros

Ajusta los parámetros de la cuadrícula de prueba en el diálogo de configuración:

![Configuración de Cuadrícula de Prueba de Material](/screenshots/material-test-grid.png)

#### Tipo de Prueba

- **Grabar:** Rellena cuadrados con patrón raster
- **Cortar:** Corta el contorno de los cuadrados

#### Rango de Velocidad

- **Velocidad Mín:** Velocidad más lenta a probar (mm/min)
- **Velocidad Máx:** Velocidad más rápida a probar (mm/min)
- Las columnas en la cuadrícula representan diferentes velocidades

#### Rango de Potencia

- **Potencia Mín:** Potencia más baja a probar (%)
- **Potencia Máx:** Potencia más alta a probar (%)
- Las filas en la cuadrícula representan diferentes niveles de potencia

#### Dimensiones de la Cuadrícula

- **Columnas:** Número de variaciones de velocidad (típicamente 3-7)
- **Filas:** Número de variaciones de potencia (típicamente 3-7)

#### Tamaño y Espaciado

- **Tamaño de Forma:** Tamaño de cada cuadrado de prueba en mm (por defecto: 20mm)
- **Espaciado:** Espacio entre cuadrados en mm (por defecto: 5mm)

#### Etiquetas

- **Incluir Etiquetas:** Habilitar/deshabilitar etiquetas de ejes mostrando valores de velocidad y potencia
- Las etiquetas aparecen en los bordes izquierdo y superior
- Las etiquetas se graban al 10% de potencia, 1000 mm/min

### Paso 4: Generar la Cuadrícula

Haz clic en **Generar** para crear el patrón de prueba. La cuadrícula aparece en tu lienzo como una pieza de trabajo especial.

## Entendiendo el Diseño de la Cuadrícula

### Organización de la Cuadrícula

```
Potencia (%)     Velocidad (mm/min) →
    ↓         1000   2500   5000   7500   10000
  100%      [  ]   [  ]   [  ]   [  ]   [  ]
   75%      [  ]   [  ]   [  ]   [  ]   [  ]
   50%      [  ]   [  ]   [  ]   [  ]   [  ]
   25%      [  ]   [  ]   [  ]   [  ]   [  ]
   10%      [  ]   [  ]   [  ]   [  ]   [  ]
```

- **Columnas:** La velocidad aumenta de izquierda a derecha
- **Filas:** La potencia aumenta de abajo hacia arriba
- **Etiquetas:** Muestran valores exactos para cada fila/columna

### Cálculo del Tamaño de Cuadrícula

**Sin etiquetas:**

- Ancho = columnas × (tamaño_forma + espaciado) - espaciado
- Alto = filas × (tamaño_forma + espaciado) - espaciado

**Con etiquetas:**

- Añadir margen de 15mm a la izquierda y arriba para espacio de etiquetas

**Ejemplo:** Cuadrícula 5×5 con cuadrados de 20mm y espaciado de 5mm:

- Sin etiquetas: 120mm × 120mm
- Con etiquetas: 135mm × 135mm

## Orden de Ejecución (Optimización de Riesgo)

Rayforge ejecuta las celdas de prueba en un **orden optimizado por riesgo** para prevenir daño al material:

1. **Velocidad más alta primero:** Las velocidades rápidas son más seguras (menor acumulación de calor)
2. **Menor potencia dentro de cada velocidad:** Minimiza el riesgo en cada nivel de velocidad

Esto previene chamuscado o fuego al comenzar con combinaciones lentas y de alta potencia.

**Ejemplo de orden de ejecución para cuadrícula 3×3:**

```
Orden:  1  2  3
        4  5  6  ← Velocidad más alta, potencia aumentando
        7  8  9

(Velocidad más rápida/potencia más baja ejecutada primero)
```

## Usando Resultados de Prueba de Material

### Paso 1: Ejecutar la Prueba

1. Carga tu material en el láser
2. Enfoca el láser correctamente
3. Ejecuta el trabajo de cuadrícula de prueba de material
4. Monitorea la prueba - detén si alguna celda causa problemas

### Paso 2: Evaluar Resultados

Después de completar la prueba, examina cada celda:

- **Muy claro:** Aumenta potencia o disminuye velocidad
- **Muy oscuro/chamuscado:** Disminuye potencia o aumenta velocidad
- **Perfecto:** Anota la combinación de velocidad/potencia

### Paso 3: Registrar Ajustes

Documenta tus ajustes exitosos para referencia futura:

- Tipo y espesor de material
- Tipo de operación (grabar o cortar)
- Combinación de velocidad y potencia
- Número de pasadas
- Cualquier nota especial

:::tip Base de Datos de Materiales
Considera crear un documento de referencia con tus resultados de prueba de material para consulta rápida en proyectos futuros.
:::

## Uso Avanzado

### Combinando con Otras Operaciones

Las cuadrículas de prueba de material son piezas de trabajo normales - puedes combinarlas con otras operaciones:

**Ejemplo de flujo de trabajo:**

1. Crear cuadrícula de prueba de material
2. Añadir corte de contorno alrededor de toda la cuadrícula
3. Ejecutar prueba, liberar, evaluar resultados

Esto es útil para liberar la pieza de prueba del material base.

### Rangos de Prueba Personalizados

Para ajuste fino, crea pruebas de rango estrecho:

**Prueba gruesa** (encontrar rango aproximado):

- Velocidad: 1000-10000 mm/min (5 columnas)
- Potencia: 10-100% (5 filas)

**Prueba fina** (optimizar):

- Velocidad: 4000-6000 mm/min (5 columnas)
- Potencia: 35-45% (5 filas)

### Diferentes Materiales, Misma Cuadrícula

Ejecuta la misma configuración de cuadrícula en diferentes materiales para construir tu biblioteca de materiales más rápido.

## Consejos y Mejores Prácticas

### Diseño de Cuadrícula

✅ **Comienza con preajustes** - Buenos puntos de partida para escenarios comunes
✅ **Usa cuadrículas 5×5** - Buen balance entre detalle y tiempo de prueba
✅ **Habilita etiquetas** - Esenciales para identificar resultados
✅ **Mantén cuadrados ≥20mm** - Más fácil de ver y medir resultados

### Estrategia de Prueba

✅ **Prueba primero en material de desecho** - Nunca pruebes en material final
✅ **Una variable a la vez** - Prueba rango de velocidad O potencia, no ambos extremos
✅ **Permite enfriamiento** - Espera entre pruebas en el mismo material
✅ **Enfoque consistente** - Misma distancia de enfoque para todas las pruebas

### Seguridad

⚠️ **Monitorea las pruebas** - Nunca dejes pruebas en ejecución sin supervisión
⚠️ **Comienza conservador** - Comienza con rangos de potencia más bajos
⚠️ **Verifica ventilación** - Asegura extracción de humos adecuada
⚠️ **Vigilancia de fuego** - Ten extintor listo

## Solución de Problemas

### Las celdas de prueba ejecutan en orden incorrecto

- Rayforge usa orden optimizado por riesgo (velocidades más rápidas primero)
- Esto es intencional y no puede cambiarse
- Ver [Orden de Ejecución](#orden-de-ejecución-optimización-de-riesgo) arriba

### Los resultados son inconsistentes

- **Verifica:** El material está plano y correctamente asegurado
- **Verifica:** El enfoque es consistente en toda el área de prueba
- **Verifica:** La potencia del láser es estable (revisa la fuente de alimentación)
- **Prueba:** Cuadrícula más pequeña para reducir el área de prueba

## Temas Relacionados

- **[Modo Simulación](../simulation-mode)** - Previsualizar ejecución de prueba antes de ejecutar
- **[Grabado](engrave)** - Entender operaciones de grabado
- **[Corte de Contorno](contour)** - Entender operaciones de corte
