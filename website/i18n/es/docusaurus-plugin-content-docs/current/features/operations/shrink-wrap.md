# Envoltura Contraída

La Envoltura Contraída crea una trayectoria de corte eficiente alrededor de múltiples objetos generando un límite que se "contrae" alrededor de ellos. Es útil para cortar múltiples partes de una lámina con desperdicio mínimo.

## Resumen

Las operaciones de Envoltura Contraída:

- Crean trayectorias de límite alrededor de grupos de objetos
- Minimizan el desperdicio de material
- Reducen el tiempo de corte combinando trayectorias
- Soportan distancias de desplazamiento para holgura
- Funcionan con cualquier combinación de formas vectoriales

## Cuándo Usar Envoltura Contraída

Usa envoltura contraída para:

- Cortar múltiples partes pequeñas de una lámina
- Minimizar el desperdicio de material
- Crear límites de anidamiento eficientes
- Separar grupos de partes
- Reducir el tiempo total de corte

**No uses envoltura contraída para:**

- Objetos individuales (usa [Contorno](contour) en su lugar)
- Partes que necesitan límites individuales
- Cortes rectangulares precisos

## Cómo Funciona la Envoltura Contraída

La envoltura contraída crea un límite usando un algoritmo de geometría computacional:

1. **Comienza** con un casco convexo alrededor de todos los objetos
2. **Contrae** el límite hacia adentro hacia los objetos
3. **Envuelve** firmemente alrededor del grupo de objetos
4. **Desplaza** hacia afuera por la distancia especificada

El resultado es una trayectoria de corte eficiente que sigue la forma general de tus partes mientras mantiene la holgura.

## Creando una Operación de Envoltura Contraída

### Paso 1: Organizar Objetos

1. Coloca todas las partes que quieres envolver en el lienzo
2. Posicionales con el espaciado deseado
3. Múltiples grupos separados pueden envolverse juntos

### Paso 2: Seleccionar Objetos

1. Selecciona todos los objetos a incluir en la envoltura contraída
2. Pueden ser diferentes formas, tamaños y tipos
3. Todos los objetos seleccionados se envolverán juntos

### Paso 3: Añadir Operación de Envoltura Contraída

- **Menú:** Operaciones → Añadir Envoltura Contraída
- **Clic derecho:** Menú contextual → Añadir Operación → Envoltura Contraída

### Paso 4: Configurar Ajustes

![Ajustes de paso de envoltura contraída](/screenshots/step-settings-shrink-wrap-general.png)

## Ajustes Clave

### Potencia y Velocidad

Como otras operaciones de corte:

**Potencia (%):**

- Intensidad del láser para cortar
- Igual que usarías para corte de [Contorno](contour)

**Velocidad (mm/min):**

- Qué tan rápido se mueve el láser
- Coincide con la velocidad de corte de tu material

**Pasadas:**

- Número de veces para cortar el límite
- Usualmente 1-2 pasadas
- Igual que el corte de contorno para tu material

### Distancia de Desplazamiento

**Desplazamiento (mm):**

- Cuánta holgura alrededor de las partes
- Distancia desde los objetos hasta el límite de envoltura contraída
- Desplazamiento mayor = más material dejado alrededor de las partes

**Valores típicos:**

- **2-3mm:** Envoltura ajustada, desperdicio mínimo
- **5mm:** Holgura cómoda
- **10mm+:** Material extra para manejo

**Por qué importa el desplazamiento:**

- Demasiado pequeño: Riesgo de cortar en las partes
- Demasiado grande: Desperdicia material
- Considera: Ancho de kerf, precisión de corte

### Suavidad

Controla qué tan de cerca el límite sigue las formas de los objetos:

**Suavidad alta:**

- Sigue los objetos más de cerca
- Trayectoria más compleja
- Tiempo de corte más largo
- Menos desperdicio de material

**Suavidad baja:**

- Trayectoria más simple, más redondeada
- Tiempo de corte más corto
- Ligeramente más desperdicio de material

**Recomendado:** Suavidad media para la mayoría de casos

## Casos de Uso

### Producción de Partes por Lote

**Escenario:** Cortar 20 partes pequeñas de una lámina grande

**Sin envoltura contraída:**

- Cortar límite de lámina completa
- Desperdiciar todo el material alrededor de las partes
- Tiempo de corte largo

**Con envoltura contraída:**

- Cortar límite ajustado alrededor del grupo de partes
- Guardar material para otros proyectos
- Corte más rápido (perímetro más corto)

### Optimización de Anidamiento

**Flujo de trabajo:**

1. Anidar partes eficientemente en la lámina
2. Agrupar partes en secciones
3. Envolver contraída cada sección
4. Cortar secciones separadamente

**Beneficios:**

- Puedes remover secciones terminadas mientras continúas
- Manejo más fácil de partes cortadas
- Riesgo reducido de movimiento de partes

### Conservación de Material

**Ejemplo:** Partes pequeñas en material costoso

**Proceso:**

1. Organizar partes ajustadamente
2. Envolver contraída con desplazamiento de 3mm
3. Cortar libre de la lámina
4. Guardar el material restante

**Resultado:** Máxima eficiencia de material

## Combinando con Otras Operaciones

### Envoltura Contraída + Contorno

Flujo de trabajo común:

1. Operaciones de **Contorno** en partes individuales (cortar detalles)
2. **Envoltura contraída** alrededor del grupo (cortar libre de la lámina)

**Orden de ejecución:**

- Primero: Cortar detalles en las partes (mientras están aseguradas)
- Último: Envoltura contraída corta el grupo libre

Ver [Flujo de Trabajo Multi-Capa](../multi-layer) para detalles.

### Envoltura Contraída + Rasterizado

**Ejemplo:** Partes grabadas y cortadas

1. **Rasterizado** graba logos en las partes
2. **Contorno** corta contornos de las partes
3. **Envoltura contraída** alrededor de todo el grupo

**Beneficios:**

- Todo el grabado ocurre mientras el material está asegurado
- La envoltura contraída final corta todo el lote libre

## Consejos y Mejores Prácticas

![Ajustes de post-procesamiento de envoltura contraída](/screenshots/step-settings-shrink-wrap-post.png)

### Espaciado de Partes

**Espaciado óptimo:**

- 5-10mm entre partes
- Suficiente para que la envoltura contraída distinga objetos separados
- No tanto que desperdicies material

**Demasiado cerca:**

- Las partes pueden envolverse juntas
- La envoltura contraída puede tender puentes sobre huecos
- Difícil de separar después de cortar

**Demasiado lejos:**

- Desperdicia material
- Tiempo de corte más largo
- Uso ineficiente de la lámina

### Consideraciones de Material

**Mejor para:**

- Tandas de producción (muchas partes idénticas)
- Partes pequeñas de láminas grandes
- Materiales costosos (minimizar desperdicio)
- Trabajos de corte por tanda

**No ideal para:**

- Partes grandes individuales
- Partes que llenan toda la lámina
- Cuando necesitas corte de lámina completa

### Seguridad

**Siempre:**

- Verifica que el límite no se superponga con las partes
- Verifica que el desplazamiento sea suficiente
- Previsualiza en [Modo Simulación](../simulation-mode)
- Prueba en desecho primero

**Busca:**

- Envoltura contraída cortando en las partes (aumenta desplazamiento)
- Partes moviéndose antes de que la envoltura contraída se complete
- Material alabeándose sacando partes de posición

## Técnicas Avanzadas

### Múltiples Envolturas Contraídas

Crea límites separados para diferentes grupos:

**Proceso:**

1. Organizar partes en grupos lógicos
2. Envolver contraída Grupo 1 (partes superiores)
3. Envolver contraída Grupo 2 (partes inferiores)
4. Cortar grupos separadamente

**Beneficios:**

- Remover grupos terminados durante el trabajo
- Mejor organización
- Recuperación de partes más fácil

### Envolturas Contraídas Anidadas

Envoltura contraída dentro de un límite más grande:

**Ejemplo:**

1. Envoltura contraída interior: Partes detalladas pequeñas
2. Envoltura contraída exterior: Incluye partes más grandes
3. Contorno: Límite de lámina completa

**Usar para:** Diseños complejos de múltiples partes

### Prueba de Holgura

Antes de la tanda de producción:

1. Crear envoltura contraída
2. Previsualizar con [Modo Simulación](../simulation-mode)
3. Verificar que la holgura es adecuada
4. Revisar que ninguna parte esté intersectada
5. Ejecutar prueba en material de desecho

## Solución de Problemas

### La envoltura contraída corta en las partes

- **Aumenta:** Distancia de desplazamiento
- **Revisa:** Las partes no están demasiado juntas
- **Verifica:** Trayectoria de envoltura contraída en la vista previa
- **Ten en cuenta:** Ancho de kerf (ancho del haz láser)

### El límite no sigue las formas

- **Aumenta:** Ajuste de suavidad
- **Revisa:** Las partes están correctamente seleccionadas
- **Prueba:** Desplazamiento más pequeño (puede estar envolviendo demasiado hacia afuera)

### Las partes se envuelven juntas

- **Aumenta:** Espaciado entre partes
- **Añade:** Contornos manuales alrededor de partes individuales
- **Divide:** En múltiples operaciones de envoltura contraída

### El corte toma demasiado tiempo

- **Disminuye:** Suavidad (trayectoria más simple)
- **Aumenta:** Desplazamiento (límites más rectos)
- **Considera:** Múltiples envolturas contraídas más pequeñas

### Las partes se mueven durante el corte

- **Añade:** Pestañas pequeñas para sostener las partes (ver [Pestañas de Sujeción](../holding-tabs))
- **Usa:** Orden de corte: de adentro hacia afuera
- **Asegúrate:** El material está plano y asegurado
- **Revisa:** La lámina no está alabeada

## Detalles Técnicos

### Algoritmo

La envoltura contraída usa geometría computacional:

1. **Casco convexo** - Encuentra el límite exterior
2. **Forma alfa** - Contrae hacia los objetos
3. **Desplazamiento** - Expande por la distancia de desplazamiento
4. **Simplifica** - Basándose en el ajuste de suavidad

### Optimización de Trayectoria

La trayectoria del límite se optimiza para:

- Longitud total mínima
- Curvas suaves (basándose en la suavidad)
- Puntos de inicio/fin eficientes

### Sistema de Coordenadas

- **Unidades:** Milímetros (mm)
- **Precisión:** 0.01mm típico
- **Coordenadas:** Igual que el espacio de trabajo

## Temas Relacionados

- **[Corte de Contorno](contour)** - Cortar contornos de objetos individuales
- **[Flujo de Trabajo Multi-Capa](../multi-layer)** - Combinando operaciones efectivamente
- **[Pestañas de Sujeción](../holding-tabs)** - Mantener partes aseguradas durante el corte
- **[Modo Simulación](../simulation-mode)** - Previsualizando trayectorias de corte
- **[Cuadrícula de Prueba de Materiales](material-test-grid)** - Encontrar ajustes de corte óptimos
