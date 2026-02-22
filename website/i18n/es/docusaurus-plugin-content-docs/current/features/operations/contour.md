# Corte de Contorno

El corte de contorno traza el contorno de formas vectoriales para liberarlas del material. Es la operación láser más común para crear piezas, letreros y piezas decorativas.

## Resumen

Las operaciones de contorno:

- Siguen trayectorias vectoriales (líneas, curvas, formas)
- Cortan a lo largo del perímetro de los objetos
- Soportan pasadas simples o múltiples para materiales gruesos
- Pueden usar trayectorias de corte interiores, exteriores o en línea
- Funcionan con cualquier forma vectorial cerrada o abierta


## Cuándo Usar Contorno

Usa corte de contorno para:

- Liberar piezas del material base
- Crear contornos y bordes
- Cortar formas de madera, acrílico, cartón
- Perforar o marcar (con potencia reducida)
- Crear plantillas y patrones

**No uses contorno para:**

- Rellenar áreas (usa [Grabado](engrave) en su lugar)
- Imágenes bitmap (convierte a vectores primero)

## Crear una Operación de Contorno

### Paso 1: Seleccionar Objetetos

1. Importa o dibuja formas vectoriales en el lienzo
2. Selecciona los objetos que quieres cortar
3. Asegura que las formas sean trayectorias cerradas para cortes completos

### Paso 2: Añadir Operación de Contorno

- **Menú:** Operaciones Añadir Contorno
- **Atajo:** <kbd>ctrl+shift+c</kbd>
- **Clic derecho:** Menú contextual Añadir Operación Contorno

### Paso 3: Configurar Ajustes

![Configuración de contorno](/screenshots/step-settings-contour-general.png)

## Ajustes Principales

### Potencia y Velocidad

**Potencia (%):**

- Intensidad del láser de 0-100%
- Mayor potencia para materiales más gruesos
- Menor potencia para marcar o puntuar

**Velocidad (mm/min):**

- Qué tan rápido se mueve el láser
- Más lento = más energía = corte más profundo
- Más rápido = menos energía = corte más ligero

### Corte Multi-Pasada

Para materiales más gruesos de lo que una sola pasada puede cortar:

**Pasadas:**

- Número de veces que se repite el corte
- Cada pasada corta más profundo

**Profundidad de Pasada (paso Z):**

- Cuánto bajar el eje Z por pasada (si es soportado)
- Requiere control de eje Z en tu máquina
- Crea corte 2.5D verdadero
- Configura en 0 para pasadas múltiples a la misma profundidad

:::warning Eje Z Requerido
:::

La profundidad de pasada solo funciona si tu máquina tiene control de eje Z. Para máquinas sin eje Z, usa pasadas múltiples a la misma profundidad.

### Desplazamiento de Trayectoria

Controla dónde corta el láser en relación a la trayectoria vectorial:

| Desplazamiento  | Descripción                 | Usar Para                        |
| --------------- | --------------------------- | -------------------------------- |
| **En Línea**    | Corta directamente en la trayectoria | Cortes de línea central, marcado |
| **Interior**    | Corta dentro de la forma    | Piezas que deben ajustarse al tamaño exacto |
| **Exterior**    | Corta fuera de la forma     | Agujeros donde las piezas se ajustan |

**Distancia de Desplazamiento:**

- Qué tan lejos dentro/fuera desplazar (mm)
- Típicamente configurado a la mitad del ancho de tu kerf
- Kerf = ancho de material removido por el láser
- Ejemplo: 0.15mm de desplazamiento para 0.3mm de kerf

### Dirección de Corte

**Horario vs Antihorario:**

- Afecta qué lado del corte recibe más calor
- Generalmente horario para regla de la mano derecha
- Cambia si un lado se quema más que el otro

**Optimizar Orden:**

- Ordena automáticamente trayectorias para viaje mínimo
- Reduce tiempo de trabajo
- Previene cortes perdidos

## Funciones Avanzadas

![Configuración de post-procesamiento de contorno](/screenshots/step-settings-contour-post.png)

### Pestañas de Sujeción

Las pestañas mantienen las piezas cortadas adjuntas al material base durante el corte:

- Añade pestañas para prevenir que las piezas caigan
- Las pestañas son pequeñas secciones sin cortar
- Rompe las pestañas después de completar el trabajo
- Ver [Pestañas de Sujeción](../holding-tabs) para detalles

### Compensación de Kerf

Kerf es el ancho de material removido por el haz láser:

**Por qué importa:**

- Un círculo cortado "en línea" será ligeramente más pequeño que el diseño
- El láser remueve ~0.2-0.4mm de material (dependiendo del ancho del haz)

**Cómo compensar:**

1. Mide tu kerf en cortes de prueba
2. Usa desplazamiento de trayectoria = kerf/2
3. Para piezas: desplaza **hacia adentro** por kerf/2
4. Para agujeros: desplaza **hacia afuera** por kerf/2

Ver [Kerf](../kerf) para guía detallada.

### Entrada/Salida

Las entradas y salidas controlan dónde comienzan y terminan los cortes:

**Entrada:**

- Entrada gradual al corte
- Previene marcas de quemadura en el punto de inicio
- Mueve el láser a velocidad completa antes de alcanzar el borde del material

**Salida:**

- Salida gradual del corte
- Previene daño en el punto final
- Común para metales y acrílicos

**Configuración:**

- Longitud: Qué tan lejos se extiende la entrada (mm)
- Ángulo: Dirección de la trayectoria de entrada
- Tipo: Línea recta, arco o espiral

## Consejos y Mejores Prácticas

### Prueba de Material

**Siempre prueba primero:**

1. Corta pequeñas formas de prueba en material de desecho
2. Comienza con ajustes conservadores (menor potencia, menor velocidad)
3. Gradualmente aumenta potencia o disminuye velocidad
4. Registra los ajustes exitosos

### Orden de Corte

**Mejores prácticas:**

- Graba antes de cortar (mantiene el material asegurado)
- Corta características interiores antes del perímetro exterior
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

- **Añade:** [Pestañas de sujeción](../holding-tabs)
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
- **Origen:** Depende de la máquina y configuración del trabajo
- **Coordenadas:** Plano X/Y (Z para profundidad multi-pasada)

### Generación de Trayectoria

Rayforge convierte formas vectoriales a G-code:

1. Desplazar trayectoria (si es corte interior/exterior)
2. Optimizar orden de trayectoria (minimizar viaje)
3. Insertar entrada/salida (si está configurado)
4. Añadir pestañas de sujeción (si están configuradas)
5. Generar comandos G-code

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
- **[Pestañas de Sujeción](../holding-tabs)** - Mantener piezas aseguradas durante el corte
- **[Kerf](../kerf)** - Mejorar precisión de corte
- **[Cuadrícula de Prueba de Material](material-test-grid)** - Encontrar ajustes óptimos de potencia/velocidad
