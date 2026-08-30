# Envoltura Contraída

La Envoltura Contraída crea una trayectoria de corte eficiente alrededor de múltiples objetos generando un límite que se "contrae" alrededor de ellos. Es útil para cortar múltiples piezas de una lámina con desperdicio mínimo.

## Resumen

Las operaciones de Envoltura Contraída:

- Crean trayectorias de límite alrededor de grupos de objetos
- Minimizan el desperdicio de material
- Reducen el tiempo de corte combinando trayectorias
- Soportan distancias de desplazamiento para holgura
- Funcionan con cualquier combinación de formas vectoriales

## Cuándo Usar Envoltura Contraída

Usa la envoltura contraída para:

- Cortar múltiples piezas pequeñas de una lámina
- Minimizar el desperdicio de material
- Crear límites de anidamiento eficientes
- Separar grupos de piezas
- Reducir el tiempo total de corte

**No uses envoltura contraída para:**

- Objetos individuales (usa [Contorno](contour) en su lugar)
- Piezas que necesitan límites individuales
- Cortes rectangulares precisos

## Cómo Funciona la Envoltura Contraída

La envoltura contraída crea un límite usando un algoritmo de geometría computacional:

1. **Comienza** con un casco convexo alrededor de todos los objetos
2. **Contrae** el límite hacia adentro, hacia los objetos
3. **Envuelve** firmemente alrededor del grupo de objetos
4. **Desplaza** hacia afuera por la distancia especificada

El resultado es una trayectoria de corte eficiente que sigue la forma general de tus piezas mientras mantiene la holgura.

## Crear una Operación de Envoltura Contraída

### Paso 1: Organizar Objetos

1. Coloca todas las piezas que quieres envolver en el lienzo
2. Posiciónalas con el espaciado deseado
3. Múltiples grupos separados pueden envolverse juntos

### Paso 2: Seleccionar Objetos

1. Selecciona todos los objetos a incluir en la envoltura contraída
2. Pueden ser diferentes formas, tamaños y tipos
3. Todos los objetos seleccionados se envolverán juntos

### Paso 3: Añadir Operación de Envoltura Contraída

- **Menú:** Operaciones Añadir Envoltura Contraída
- **Clic derecho:** Menú contextual Añadir Operación Envoltura Contraída

### Paso 4: Configurar Ajustes

![Ajustes de paso de envoltura contraída](/screenshots/step-settings-shrink-wrap-general.webp)

## Ajustes Clave

El diálogo de ajustes de paso tiene tres pestañas: **Ajustes de Paso**, **Láser** y **Post-Procesamiento**. Los ajustes se describen en orden de pestañas a continuación.

### Envoltura Contraída

El grupo **Envoltura Contraída** en la pestaña *Ajustes de Paso* controla cómo se ajusta el casco alrededor del contenido.

#### Suavidad

Controla qué tan de cerca el límite sigue las formas de los objetos:

**Suavidad alta:**

- Sigue los objetos más de cerca
- Trayectoria más compleja
- Tiempo de corte más largo
- Menos desperdicio de material

**Suavidad baja:**

- Trayectoria más simple y redondeada
- Tiempo de corte más corto
- Ligeramente más desperdicio de material

**Recomendado:** Suavidad media para la mayoría de los casos

#### Lado de Corte

Controla dónde corta el láser en relación a la trayectoria de envoltura contraída:

| Lado de Corte | Descripción                 | Usar Para                           |
| ------------- | --------------------------- | ----------------------------------- |
| **Línea central** | Corta directamente en la trayectoria | Corte estándar                |
| **Exterior**  | Corta fuera del límite      | Hace el corte ligeramente más grande |
| **Interior**  | Corta dentro del límite     | Hace el corte ligeramente más pequeño |

#### Distancia de Desplazamiento

**Desplazamiento (mm):**

- Cuánta holgura alrededor de las piezas
- Distancia desde los objetos hasta el límite de envoltura contraída
- Desplazamiento mayor = más material dejado alrededor de las piezas

**Valores típicos:**

- **2-3mm:** Envoltura ajustada, desperdicio mínimo
- **5mm:** Holgura cómoda
- **10mm+:** Material extra para manipulación

**Por qué importa el desplazamiento:**

- Demasiado pequeño: Riesgo de cortar dentro de las piezas
- Demasiado grande: Desperdicia material
- Considera: Ancho de kerf, precisión de corte

### Ajustes del Láser

![Ajustes del láser](/screenshots/step-settings-shrink-wrap-laser.webp)

La potencia, la velocidad y la selección del cabezal láser se encuentran en la página **Láser** del diálogo de ajustes de paso.

Como otras operaciones de corte:

**Potencia (%):**

- Intensidad del láser para cortar
- Igual que la que usarías para el corte de [Contorno](contour)

**Velocidad (mm/min):**

- Qué tan rápido se mueve el láser
- Coincide con la velocidad de corte de tu material

Para cortar el límite más de una vez, añade un post-procesador [Multi-Pasada](../multi-pass.md).

## Casos de Uso

### Producción de Piezas por Lote

**Escenario:** Cortar 20 piezas pequeñas de una lámina grande

**Sin envoltura contraída:**

- Cortar el límite de la lámina completa
- Desperdiciar todo el material alrededor de las piezas
- Tiempo de corte largo

**Con envoltura contraída:**

- Cortar un límite ajustado alrededor del grupo de piezas
- Ahorrar material para otros proyectos
- Corte más rápido (perímetro más corto)

### Optimización de Anidamiento

**Flujo de trabajo:**

1. Anidar piezas eficientemente en la lámina
2. Agrupar piezas en secciones
3. Envolver contraída cada sección
4. Cortar las secciones por separado

**Beneficios:**

- Puedes remover las secciones terminadas mientras continúas
- Manejo más fácil de las piezas cortadas
- Riesgo reducido de movimiento de las piezas

### Conservación de Material

**Ejemplo:** Piezas pequeñas en material costoso

**Proceso:**

1. Organizar las piezas ajustadamente
2. Envolver contraída con 3mm de desplazamiento
3. Cortar libre de la lámina
4. Guardar el material restante

**Resultado:** Máxima eficiencia de material

## Combinando con Otras Operaciones

### Envoltura Contraída + Contorno

Flujo de trabajo común:

1. Operaciones de **Contorno** en piezas individuales (cortar detalles)
2. **Envoltura contraída** alrededor del grupo (cortar libre de la lámina)

**Orden de ejecución:**

- Primero: Cortar detalles en las piezas (mientras están aseguradas)
- Último: La envoltura contraída corta el grupo libre

Ver [Flujo de Trabajo Multi-Capa](../multi-layer.md) para detalles.

### Envoltura Contraída + Rasterizado

**Ejemplo:** Piezas grabadas y cortadas

1. **Rasterizado** graba logos en las piezas
2. **Contorno** corta los contornos de las piezas
3. **Envoltura contraída** alrededor de todo el grupo

**Beneficios:**

- Todo el grabado ocurre mientras el material está asegurado
- La envoltura contraída final corta todo el lote libre

## Post-Procesamiento

![Ajustes de post-procesamiento de envoltura contraída](/screenshots/step-settings-shrink-wrap-post.webp)

Las operaciones de Envoltura Contraída soportan varias opciones de post-procesamiento:

- **[Suavizar Trayectoria](../smooth.md)** - Reduce bordes irregulares en la trayectoria del límite
- **[Pestañas de Sujeción](../holding-tabs.md)** - Mantienen las piezas cortadas adjuntas al material base
- **[Recortar al Material](../crop-to-stock.md)** - Limita los cortes al límite del material
- **[Optimización de Trayectoria](../path-optimization.md)** - Reduce la distancia de viaje
- **[Multi-Pasada](../multi-pass.md)** - Repite cortes para materiales gruesos
- **[Entrada/Salida](../lead-in-out.md)** - Añade movimientos de aproximación y salida sin potencia para extremos de corte más limpios

### Espaciado de Piezas

**Espaciado óptimo:**

- 5-10mm entre piezas
- Suficiente para que la envoltura contraída distinga objetos separados
- No tanto que desperdicies material

**Demasiado cerca:**

- Las piezas pueden envolverse juntas
- La envoltura contraída puede tender puentes sobre huecos
- Difícil de separar después del corte

**Demasiado lejos:**

- Desperdicia material
- Tiempo de corte más largo
- Uso ineficiente de la lámina

### Consideraciones de Material

**Mejor para:**

- Tandas de producción (muchas piezas idénticas)
- Piezas pequeñas de láminas grandes
- Materiales costosos (minimizar desperdicio)
- Trabajos de corte por tandas

**No ideal para:**

- Piezas grandes individuales
- Piezas que llenan toda la lámina
- Cuando necesitas el corte de la lámina completa

### Seguridad

**Siempre:**

- Verifica que el límite no se superponga con las piezas
- Verifica que el desplazamiento sea suficiente
- Previsualiza en [Vista Previa 3D](../../ui/3d-preview.md)
- Prueba en desecho primero

**Ten en cuenta:**

- La envoltura contraída cortando dentro de las piezas (aumenta el desplazamiento)
- Piezas que se mueven antes de que la envoltura contraída se complete
- El alabeo del material sacando piezas de posición

## Técnicas Avanzadas

### Múltiples Envolturas Contraídas

Crea límites separados para diferentes grupos:

**Proceso:**

1. Organiza las piezas en grupos lógicos
2. Envuelve contraída el Grupo 1 (piezas superiores)
3. Envuelve contraída el Grupo 2 (piezas inferiores)
4. Corta los grupos por separado

**Beneficios:**

- Remueve los grupos terminados durante el trabajo
- Mejor organización
- Recuperación de piezas más fácil

### Envolturas Contraídas Anidadas

Envoltura contraída dentro de un límite más grande:

**Ejemplo:**

1. Envoltura contraída interior: Piezas detalladas pequeñas
2. Envoltura contraída exterior: Incluye piezas más grandes
3. Contorno: Límite de la lámina completa

**Usar para:** Diseños complejos de múltiples piezas

### Prueba de Holgura

Antes de la tanda de producción:

1. Crear la envoltura contraída
2. Previsualizar con [Vista Previa 3D](../../ui/3d-preview.md)
3. Verificar que la holgura sea adecuada
4. Revisar que ninguna pieza esté intersectada
5. Ejecutar una prueba en material de desecho

## Solución de Problemas

### La envoltura contraída corta dentro de las piezas

- **Aumenta:** Distancia de desplazamiento
- **Revisa:** Las piezas no están demasiado juntas
- **Verifica:** Trayectoria de envoltura contraída en la vista previa
- **Ten en cuenta:** Ancho de kerf (ancho del haz láser)

### El límite no sigue las formas

- **Aumenta:** Ajuste de suavidad
- **Revisa:** Las piezas están seleccionadas correctamente
- **Prueba:** Desplazamiento más pequeño (puede estar envolviendo demasiado hacia afuera)

### Las piezas se envuelven juntas

- **Aumenta:** Espaciado entre piezas
- **Añade:** Contornos manuales alrededor de piezas individuales
- **Divide:** En múltiples operaciones de envoltura contraída

### El corte toma demasiado tiempo

- **Disminuye:** Suavidad (trayectoria más simple)
- **Aumenta:** Desplazamiento (límites más rectos)
- **Considera:** Múltiples envolturas contraídas más pequeñas

### Las piezas se mueven durante el corte

- **Añade:** Pestañas pequeñas para sostener las piezas (ver [Pestañas de Sujeción](../holding-tabs.md))
- **Usa:** Orden de corte: de adentro hacia afuera
- **Asegúrate:** El material está plano y asegurado
- **Revisa:** La lámina no está alabeada

## Detalles Técnicos

### Algoritmo

La envoltura contraída usa geometría computacional:

1. **Casco convexo** - Encuentra el límite exterior
2. **Forma alfa** - Contrae hacia los objetos
3. **Desplazamiento** - Expande por la distancia de desplazamiento
4. **Simplificación** - Basada en el ajuste de suavidad

### Optimización de Trayectoria

La trayectoria del límite se optimiza para:

- Longitud total mínima
- Curvas suaves (basadas en la suavidad)
- Puntos de inicio/fin eficientes

### Sistema de Coordenadas

- **Unidades:** Milímetros (mm)
- **Precisión:** 0.01mm típico
- **Coordenadas:** Igual que el espacio de trabajo

## Temas Relacionados

- **[Corte de Contorno](contour)** - Cortar contornos de objetos individuales
- **[Flujo de Trabajo Multi-Capa](../multi-layer.md)** - Combinando operaciones efectivamente
- **[Pestañas de Sujeción](../holding-tabs.md)** - Mantener piezas aseguradas durante el corte
- **[Vista Previa 3D](../../ui/3d-preview.md)** - Previsualizando trayectorias de corte
- **[Cuadrícula de Prueba de Material](material-test-grid)** - Encontrar ajustes de corte óptimos
