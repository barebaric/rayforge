# Perfilado de Marco

El Perfilado de Marco crea una trayectoria de corte rectangular simple alrededor de todo tu diseño. Es la forma más rápida de añadir un borde limpio o liberar tu trabajo de la lámina de material.

## Resumen

Las operaciones de Perfilado de Marco:

- Crean un límite rectangular alrededor de todo el contenido
- Añaden un desplazamiento/margen configurable desde el diseño
- Soportan compensación de kerf para un dimensionado preciso
- Funcionan con cualquier combinación de objetos en el lienzo

![Configuración de Perfilado de Marco](/screenshots/step-settings-frame-outline-general.png)

## Cuándo Usar Perfilado de Marco

Usa perfilado de marco para:

- Añadir un borde decorativo alrededor de tu diseño
- Liberar tu trabajo de la lámina de material
- Crear un límite rectangular simple
- Enmarcado rápido sin cálculos complejos de trayectoria

**No uses perfilado de marco para:**

- Formas irregulares alrededor de múltiples objetos (usa [Envoltura Ajustada](shrink-wrap) en su lugar)
- Cortar piezas individuales (usa [Contorno](contour) en su lugar)
- Seguir la forma exacta de tu diseño

## Crear una Operación de Perfilado de Marco

### Paso 1: Organiza Tu Diseño

1. Coloca todos los objetos en el lienzo
2. Posiciónalos donde quieras en relación al marco
3. El marco se calculará alrededor del cuadro delimitador de todo el contenido

### Paso 2: Añadir Operación de Perfilado de Marco

- **Menú:** Operaciones → Añadir Perfilado de Marco
- **Clic derecho:** Menú contextual → Añadir Operación → Perfilado de Marco

### Paso 3: Configurar Ajustes

Configura los parámetros del marco:

- **Potencia y Velocidad:** Coincide con los requisitos de corte de tu material
- **Desplazamiento:** Distancia desde el borde del contenido hasta el marco
- **Desplazamiento de Trayectoria:** Corte interior, exterior o línea central

## Ajustes Principales

### Potencia y Velocidad

**Potencia (%):**

- Intensidad del láser para cortar el marco
- Coincide con los requisitos de corte de tu material

**Velocidad (mm/min):**

- Qué tan rápido se mueve el láser
- Más lento para materiales más gruesos

**Pasadas:**

- Número de veces que se corta el marco
- Generalmente 1-2 pasadas
- Añade pasadas para materiales más gruesos

### Distancia de Desplazamiento

**Desplazamiento (mm):**

- Distancia desde el cuadro delimitador del diseño hasta el marco
- Crea un margen/borde alrededor de tu trabajo

**Valores típicos:**

- **0mm:** El marco toca el borde del diseño
- **2-5mm:** Pequeño margen para apariencia limpia
- **10mm+::** Borde grande para montaje o manipulación

### Desplazamiento de Trayectoria (Lado de Corte)

Controla dónde corta el láser en relación a la trayectoria del marco:

| Lado de Corte   | Descripción                   | Usar Para                         |
| --------------- | ----------------------------- | --------------------------------- |
| **Línea central** | Corta directamente en la trayectoria | Corte estándar                    |
| **Exterior**    | Corta fuera de la trayectoria del marco | Hace el marco ligeramente más grande |
| **Interior**    | Corta dentro de la trayectoria del marco | Hace el marco ligeramente más pequeño |

### Compensación de Kerf

El perfilado de marco soporta compensación de kerf:

- Ajusta automáticamente el ancho del haz del láser
- Asegura dimensiones finales precisas
- Usa el valor de kerf de la configuración de tu cabezal láser

## Opciones de Post-Procesamiento

![Configuración de post-procesamiento de Perfilado de Marco](/screenshots/step-settings-frame-outline-post.png)

### Multi-Pasada

Corta el marco múltiples veces:

- **Pasadas:** Número de repeticiones
- **Descenso Z:** Bajar Z entre pasadas (requiere eje Z)
- Útil para materiales gruesos

### Pestañas de Sujeción

Añade pestañas para mantener la pieza enmarcada adjunta:

- Previene que las piezas caigan durante el corte
- Configura ancho, altura y espaciado de pestañas
- Ver [Pestañas de Sujeción](../holding-tabs) para detalles

## Casos de Uso

### Borde Decorativo

**Escenario:** Añadir un borde rectangular limpio alrededor de una placa o letrero

**Proceso:**

1. Diseña tu contenido (texto, logos, etc.)
2. Añade Perfilado de Marco con 3-5mm de desplazamiento
3. Corta con ajustes de marcado decorativo (baja potencia)

**Resultado:** Pieza con borde de apariencia profesional

### Liberar de la Lámina

**Escenario:** Remover tu trabajo terminado de la lámina de material

**Proceso:**

1. Completa todas las otras operaciones (grabado, cortes de contorno)
2. Añade Perfilado de Marco como la última operación
3. Configura el desplazamiento para incluir un pequeño margen

**Beneficios:**

- Separación limpia de la lámina
- Calidad de borde consistente
- Fácil de ejecutar como paso final

### Límite para Procesamiento por Lotes

**Escenario:** Crear un límite de corte para múltiples piezas anidadas

**Proceso:**

1. Organiza todas las piezas en el lienzo
2. Añade operaciones de contorno individuales para las piezas
3. Añade Perfilado de Marco alrededor de todo
4. El marco corta al último (en capa separada)

**Orden:** Grabado → Contornos de piezas → Perfilado de marco

## Consejos y Mejores Prácticas

### Orden de Capas

**Mejor práctica:**

- Coloca el Perfilado de Marco en su propia capa
- Ejecuta el marco como la **última** operación
- Esto asegura que todo el otro trabajo se complete primero

**¿Por qué al final?**

- El material permanece asegurado durante otras operaciones
- Previene que las piezas se muevan
- Resultado final más limpio

### Selección de Desplazamiento

**Elegir desplazamiento:**

- **0-2mm:** Ajuste apretado, mínimo desperdicio de material
- **3-5mm:** Margen estándar, aspecto profesional
- **10mm+:** Material extra para manipulación/montaje

**Considera:**

- El uso final de la pieza
- Si los bordes serán visibles
- Costo y disponibilidad del material

### Ajustes de Calidad

**Para cortes de marco limpios:**

- Usa asistencia de aire
- Asegura el enfoque correcto
- Múltiples pasadas más rápidas a menudo mejor que una pasada lenta
- Mantén el material plano y asegurado

## Combinando con Otras Operaciones

### Marco + Grabado + Contorno

Flujo de trabajo típico para una pieza terminada:

1. **Capa 1:** Grabar detalles (texto, imágenes)
2. **Capa 2:** Cortar contorno de piezas individuales
3. **Capa 3:** Perfilado de marco (liberar)

**El orden de ejecución asegura:**

- El grabado ocurre mientras el material está plano y asegurado
- Los detalles de las piezas se cortan antes de la separación final
- El marco libera todo al final

### Marco vs Envoltura Ajustada

| Característica   | Perfilado de Marco           | Envoltura Ajustada      |
| ---------------- | ---------------------------- | ----------------------- |
| **Forma**        | Siempre rectangular          | Sigue contornos de objetos |
| **Velocidad**    | Muy rápida (4 líneas)        | Depende de la complejidad |
| **Caso de uso**  | Bordes simples, liberar      | Uso eficiente de material |
| **Flexibilidad** | Rectángulo fijo              | Se adapta al diseño     |

**Elige Perfilado de Marco cuando:**

- Quieres un borde rectangular
- Se prefiere simplicidad
- Liberando de la lámina

**Elige Envoltura Ajustada cuando:**

- Quieres minimizar desperdicio de material
- El diseño tiene forma irregular
- La eficiencia es importante

## Solución de Problemas

### El marco está muy apretado/suelto

- **Ajusta:** Configuración de distancia de desplazamiento
- **Verifica:** Desplazamiento de trayectoria (interior/exterior/línea central)
- **Confirma:** La compensación de kerf es correcta

### El marco no aparece

- **Verifica:** Los objetos están en el lienzo
- **Confirma:** La operación está habilitada
- **Busca:** El marco puede estar fuera del área visible (aleja el zoom)

### El marco corta el diseño

- **Aumenta:** Distancia de desplazamiento
- **Verifica:** Los objetos están posicionados correctamente
- **Confirma:** El cálculo del cuadro delimitador incluye todos los objetos

### Profundidad de corte inconsistente

- **Verifica:** El material está plano
- **Confirma:** La distancia de enfoque es correcta
- **Prueba:** Múltiples pasadas a menor potencia

## Detalles Técnicos

### Cálculo del Cuadro Delimitador

El perfilado de marco usa el cuadro delimitador combinado de:

- Todas las piezas de trabajo en el lienzo
- Sus posiciones transformadas finales
- Incluyendo cualquier rotación/escala aplicada

### Generación de Trayectoria

1. Calcular cuadro delimitador combinado
2. Aplicar distancia de desplazamiento
3. Aplicar desplazamiento de trayectoria (interior/exterior/línea central)
4. Aplicar compensación de kerf
5. Generar trayectoria G-code rectangular

### Ejemplo de G-code

```gcode
G0 X5 Y5           ; Mover al inicio del marco (con desplazamiento)
M3 S200            ; Láser encendido al 80% de potencia
G1 X95 Y5 F500     ; Cortar borde inferior
G1 X95 Y95         ; Cortar borde derecho
G1 X5 Y95          ; Cortar borde superior
G1 X5 Y5           ; Cortar borde izquierdo (completar)
M5                 ; Láser apagado
```

## Temas Relacionados

- **[Corte de Contorno](contour)** - Cortar contornos de objetos individuales
- **[Envoltura Ajustada](shrink-wrap)** - Límites irregulares eficientes
- **[Pestañas de Sujeción](../holding-tabs)** - Mantener piezas seguras durante el corte
- **[Flujo de Trabajo Multi-Capa](../multi-layer)** - Organizar operaciones efectivamente
- **[Compensación de Kerf](../kerf)** - Mejorar precisión dimensional
