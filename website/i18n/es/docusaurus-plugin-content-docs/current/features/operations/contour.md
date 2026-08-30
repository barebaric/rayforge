# Corte de Contorno

El corte de contorno traza el contorno de formas vectoriales para cortarlas y liberarlas del
material. Es la operación láser más común para crear piezas, letreros y piezas decorativas.

## Resumen

Las operaciones de contorno:

- Siguen trayectorias vectoriales (líneas, curvas, formas)
- Cortan a lo largo del perímetro de los objetos
- Soportan pasadas simples o múltiples para materiales gruesos
- Pueden usar trayectorias de corte interiores, exteriores o en línea
- Funcionan con cualquier forma vectorial cerrada o abierta

## Cuándo Usar Contorno

Usa el corte de contorno para:

- Liberar piezas del material base
- Crear contornos y bordes
- Cortar formas de madera, acrílico, cartón
- Perforar o marcar (con potencia reducida)
- Crear plantillas y patrones

**No uses contorno para:**

- Rellenar áreas (usa [Grabado](engrave) en su lugar)
- Imágenes de mapa de bits (convierte a vectores primero)

## Crear una Operación de Contorno

### Paso 1: Seleccionar Objetos

1. Importa o dibuja formas vectoriales en el lienzo
2. Selecciona los objetos que quieres cortar
3. Asegúrate de que las formas sean trayectorias cerradas para cortes completos

### Paso 2: Añadir Operación de Contorno

- **Menú:** Operaciones Añadir Contorno
- **Atajo:** <kbd>ctrl+shift+c</kbd>
- **Clic derecho:** Menú contextual Añadir Operación Contorno

### Paso 3: Configurar Ajustes

![Configuración de contorno](/screenshots/step-settings-contour-general.webp)

## Ajustes Principales

El diálogo de ajustes de paso tiene tres pestañas: **Ajustes de Paso**, **Láser** y
**Post-Procesamiento**. Los ajustes se describen en orden de pestañas a continuación.

### Ajustes de Contorno

![Configuración de contorno](/screenshots/step-settings-contour-general.webp)

El grupo **Ajustes de Contorno** en la pestaña _Ajustes de Paso_ controla cómo se traza el contorno.

#### Lado de Corte y Desplazamiento de Trayectoria

Controla dónde corta el láser en relación a la trayectoria vectorial:

| Desplazamiento    | Descripción                          | Usar Para                                   |
| ----------------- | ------------------------------------ | ------------------------------------------- |
| **Línea central** | Corta directamente en la trayectoria | Cortes de línea central, marcado            |
| **Interior**      | Corta dentro de la forma             | Piezas que deben ajustarse al tamaño exacto |
| **Exterior**      | Corta fuera de la forma              | Agujeros donde se ajustan las piezas        |

**Distancia de Desplazamiento:**

- Qué tan lejos desplazar hacia adentro/fuera (mm)
- Típicamente configurado a la mitad del ancho de tu kerf
- Kerf = ancho de material removido por el láser
- Ejemplo: 0.15mm de desplazamiento para 0.3mm de kerf

#### Orden de Corte

Controla el orden en que se procesan las trayectorias anidadas:

**Interior-Exterior:**

- Corta primero las características interiores y luego trabaja hacia afuera
- Mantiene las partes exteriores del material intactas por más tiempo

**Exterior-Interior:**

- Corta primero el perímetro exterior y luego se mueve hacia adentro
- Mantiene la pieza de trabajo asegurada al material base por más tiempo

**Recomendado:** Interior-Exterior (predeterminado)

#### Eliminar Trayectorias Interiores

Para diseños con agujeros o recortes internos, puedes elegir trazar solo el límite más exterior:

- **Eliminar Trayectorias Interiores**: Cuando está habilitado, solo se traza el contorno más
  exterior
- Los agujeros y recortes internos se ignoran

Esto es útil cuando quieres cortar una forma pero preservar el interior, como crear un marco o un
contorno sin cortar detalles internos.

#### Sobrecorte

Extiende las trayectorias de corte cerradas más allá de su punto de inicio para que el rayo láser se
superponga con el inicio del corte:

**Sobrecorte:**

- Distancia en unidades de máquina para extender el corte más allá de la unión inicio/fin
- Establecer en **0** para desactivar (predeterminado)
- Valores típicos: 1–5 para la mayoría de los materiales
- Máximo: 100

**Por qué usar sobrecorte:**

Al inicio y al final de un contorno cerrado, es posible que el láser no penetre completamente debido
a la aceleración y desaceleración. El sobrecorte asegura que el haz se superponga en la unión,
creando un corte limpio y completamente separado. Esto es especialmente útil para:

- Materiales gruesos donde la penetración completa es marginal
- Cortes a alta velocidad donde los efectos de aceleración son más pronunciados
- Piezas que deben caer libres sin post-procesamiento

El sobrecorte se aplica tanto a contornos exteriores como a agujeros internos.

:::tip Entrada/Salida vs Sobrecorte [Entrada/Salida](../lead-in-out.md) agrega movimientos de
aproximación y salida con potencia cero antes y después de la trayectoria de corte. El sobrecorte
extiende la propia trayectoria de corte más allá de la unión. Pueden usarse juntos para una calidad
de corte óptima. :::

#### Re-trazado con Umbral Personalizado

Cuando trabajas con imágenes de mapa de bits convertidas a vectores, puedes controlar qué partes se
trazan:

- **Re-escanear Contenido**: Habilita un umbral de brillo personalizado para el trazado
- **Umbral de Trazado (0.0-1.0)**: Valor de corte de brillo cuando el re-escaneo está habilitado
  - Los valores más bajos trazan solo las áreas más oscuras
  - Los valores más altos incluyen las áreas más claras

Esto es útil cuando el trazado predeterminado no captura el nivel de detalle que necesitas.

### Ajustes del Láser

![Ajustes del láser](/screenshots/step-settings-contour-laser.webp)

La potencia, la velocidad y la selección del cabezal láser se encuentran en la página **Láser** del
diálogo de ajustes de paso.

#### Potencia y Velocidad

**Potencia (%):**

- Intensidad del láser de 0-100%
- Mayor potencia para materiales más gruesos
- Menor potencia para marcar o puntuar

**Velocidad (mm/min):**

- Qué tan rápido se mueve el láser
- Más lento = más energía = corte más profundo
- Más rápido = menos energía = corte más ligero

#### Compensación de Kerf

Kerf es el ancho de material removido por el haz láser:

**Por qué importa:**

- Un círculo cortado "en línea" será ligeramente más pequeño que el diseño
- El láser remueve ~0.2-0.4mm de material (dependiendo del ancho del haz)

**Cómo compensar:**

1. Mide tu kerf en cortes de prueba
2. Usa desplazamiento de trayectoria = kerf/2
3. Para piezas: desplaza **hacia adentro** por kerf/2
4. Para agujeros: desplaza **hacia afuera** por kerf/2

Ver [Kerf](../kerf.md) para una guía detallada.

## Post-Procesamiento

![Configuración de post-procesamiento de contorno](/screenshots/step-settings-contour-post.webp)

Las operaciones de contorno soportan varias opciones de post-procesamiento:

- **[Suavizar Trayectoria](../smooth.md)** - Reduce bordes irregulares en trayectorias de corte
- **[Pestañas de Sujeción](../holding-tabs.md)** - Mantienen las piezas cortadas adjuntas al
  material base
- **[Recortar al Material](../crop-to-stock.md)** - Limita los cortes al límite del material
- **[Optimización de Trayectoria](../path-optimization.md)** - Reduce la distancia de viaje entre
  cortes
- **[Multi-Pasada](../multi-pass.md)** - Repite cortes para materiales gruesos
- **[Entrada/Salida](../lead-in-out.md)** - Agrega movimientos de aproximación y salida sin potencia
  para extremos de corte más limpios

### Corte Multi-Pasada

Para materiales más gruesos de lo que una sola pasada puede cortar:

**Pasadas:**

- Número de veces que se repite el corte
- Cada pasada corta más profundo

**Profundidad de Pasada (paso Z):**

- Cuánto bajar el eje Z por pasada (si es soportado)
- Requiere control de eje Z en tu máquina
- Crea un corte 2.5D verdadero
- Configura en 0 para pasadas múltiples a la misma profundidad

:::warning Eje Z Requerido :::

La profundidad de pasada solo funciona si tu máquina tiene control de eje Z. Para máquinas sin eje
Z, usa pasadas múltiples a la misma profundidad.

## Consejos y Mejores Prácticas

### Prueba de Material

**Siempre prueba primero:**

1. Corta pequeñas formas de prueba en material de desecho
2. Comienza con ajustes conservadores (menor potencia, menor velocidad)
3. Aumenta gradualmente la potencia o disminuye la velocidad
4. Registra los ajustes exitosos

### Orden de Corte

**Mejores prácticas:**

- Graba antes de cortar (mantiene el material asegurado)
- Corta las características interiores antes del perímetro exterior
- Usa pestañas de sujeción para piezas que puedan moverse
- Corta las piezas más pequeñas primero (menos vibración)

## Solución de Problemas

### Los cortes no atraviesan el material

- **Aumenta:** Configuración de potencia
- **Disminuye:** Configuración de velocidad
- **Añade:** Más pasadas
- **Verifica:** El enfoque es correcto
- **Verifica:** El haz está limpio (lente sucio)

### Chamuscado o quemadura excesiva

- **Disminuye:** Configuración de potencia
- **Aumenta:** Configuración de velocidad
- **Usa:** Asistencia de aire
- **Prueba:** Múltiples pasadas más rápidas en lugar de una lenta
- **Verifica:** El material es apropiado para corte láser

### Las piezas caen durante el corte

- **Añade:** [Pestañas de sujeción](../holding-tabs.md)
- **Usa:** Optimización de orden de corte
- **Corta:** Características interiores antes del exterior
- **Asegura:** El material está plano y asegurado

### Profundidad de corte inconsistente

- **Verifica:** El espesor del material es uniforme
- **Verifica:** El material está plano (no deformado)
- **Verifica:** La distancia de enfoque es consistente
- **Confirma:** La potencia del láser es estable

### Esquinas o curvas perdidas

- **Disminuye:** Velocidad (especialmente en esquinas)
- **Verifica:** Ajustes de aceleración de la máquina
- **Confirma:** Las correas están tensas
- **Reduce:** Complejidad de la trayectoria (simplifica curvas)

## Detalles Técnicos

### Sistema de Coordenadas

Las operaciones de contorno trabajan en:

- **Unidades:** Milímetros (mm)
- **Origen:** Depende de la máquina y la configuración del trabajo
- **Coordenadas:** Plano X/Y (Z para profundidad multi-pasada)

### Generación de Trayectoria

Rayforge convierte formas vectoriales a G-code:

1. Desplazar la trayectoria (si es corte interior/exterior)
2. Optimizar el orden de la trayectoria (minimizar viaje)
3. Añadir pestañas de sujeción (si están configuradas)
4. Generar comandos G-code

### Comandos G-code

G-code típico de contorno:

```gcode
G0 X10 Y10          ; Movimiento rápido al inicio
M3 S204             ; Láser encendido al 80% de potencia
G1 X50 Y10 F500     ; Cortar al punto a 500 mm/min
G1 X50 Y50 F500     ; Cortar al siguiente punto
G1 X10 Y50 F500     ; Continuar cortando
G1 X10 Y10 F500     ; Completar el cuadrado
M5                  ; Láser apagado
```

## Temas Relacionados

- **[Grabado](engrave)** - Rellenar áreas con patrones de grabado
- **[Pestañas de Sujeción](../holding-tabs.md)** - Mantener piezas aseguradas durante el corte
- **[Kerf](../kerf.md)** - Mejorar la precisión de corte
- **[Cuadrícula de Prueba de Material](material-test-grid)** - Encontrar ajustes óptimos de
  potencia/velocidad
