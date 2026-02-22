# Macros y Hooks

Rayforge proporciona dos potentes funciones de automatización para personalizar tu flujo de trabajo: **Macros** y **Hooks**. Ambas te permiten inyectar código G personalizado en tus trabajos, pero sirven propósitos diferentes.

![Ajustes de Hooks y Macros](/screenshots/machine-hooks-macros.png)

---

## Resumen

| Función     | Propósito                     | Disparador           | Caso de Uso                                     |
| ----------- | ----------------------------- | -------------------- | ----------------------------------------------- |
| **Macros**  | Fragmentos de código G reutilizables | Ejecución manual     | Comandos rápidos, patrones de prueba, rutinas personalizadas |
| **Hooks**   | Inyección automática de código G | Eventos del ciclo de vida del trabajo | Secuencias de inicio, cambios de capa, limpieza |

---

## Macros

Las macros son **scripts de código G nombrados y reutilizables** que puedes ejecutar manualmente en cualquier momento.

### ¿Para Qué Sirven las Macros?

Casos de uso comunes de macros:

- **Llevar la máquina al origen** - Enviar `$H` rápidamente
- **Establecer desplazamientos de trabajo** - Almacenar y recuperar posiciones G54/G55
- **Control de asistencia de aire** - Activar/desactivar asistencia de aire
- **Prueba de enfoque** - Ejecutar un patrón de prueba de enfoque rápido
- **Cambios de herramienta personalizados** - Para configuraciones con múltiples láseres
- **Rutinas de emergencia** - Apagado rápido o limpieza de alarma
- **Sondeo de material** - Autoenfoque o medición de altura

### Creando una Macro

1. **Abrir Configuración de Máquina:**
   - Navega a **Configuración Máquina Macros**

2. **Añadir una nueva macro:**
   - Haz clic en el botón **"+"**
   - Ingresa un nombre descriptivo (ej., "Llevar Máquina al Origen", "Habilitar Asistencia de Aire")

3. **Escribe tu código G:**
   - Cada línea es un comando de código G separado
   - Los comentarios comienzan con `;` o `(`
   - Se pueden usar variables (ver Sustitución de Variables más abajo)

4. **Guarda la macro**

5. **Ejecuta la macro:**
   - Desde la lista de macros, haz clic en la macro
   - O asigna un atajo de teclado (si es compatible)

### Ejemplos de Macros

#### Simple: Llevar la Máquina al Origen

**Nombre:** Llevar Máquina al Origen
**Código:**

```gcode
$H
; Espera a que el homing se complete
```

**Uso:** Lleva rápidamente la máquina al origen antes de comenzar a trabajar.

---

#### Medio: Establecer Desplazamiento de Trabajo

**Nombre:** Establecer G54 a la Posición Actual
**Código:**

```gcode
G10 L20 P1 X0 Y0
; Establece el origen del sistema de coordenadas de trabajo G54 a la posición actual
```

**Uso:** Marca la posición actual del láser como el origen del trabajo.

---

#### Avanzado: Cuadrícula de Prueba de Enfoque

**Nombre:** Prueba de Enfoque de 9 Puntos
**Código:**

```gcode
; Cuadrícula de 9 puntos para encontrar el enfoque óptimo
G21  ; Milímetros
G90  ; Posicionamiento absoluto
G0 X10 Y10
M3 S1000
G4 P0.1
M5
G0 X20 Y10
M3 S1000
G4 P0.1
M5
; ... (repetir para los puntos restantes)
```

**Uso:** Prueba rápidamente el enfoque en diferentes posiciones de la cama.

---

---

### Ejemplos de Macros

Los Hooks son **inyecciones automáticas de código G** activadas por eventos específicos durante la ejecución del trabajo.

### Disparadores de Hooks

Rayforge soporta estos disparadores de hooks:

| Disparador            | Cuándo se Ejecuta                     | Usos Comunes                                 |
| --------------------- | ------------------------------------- | ------------------------------------------- |
| **Inicio de Trabajo** | Muy al comienzo del trabajo           | Homing, desplazamiento de trabajo, asistencia de aire on, precalentamiento |
| **Fin de Trabajo**    | Muy al final del trabajo              | Volver al origen, asistencia de aire off, pitido, enfriamiento |
| **Inicio de Capa**    | Antes de procesar cada capa           | Cambio de herramienta, ajuste de potencia, comentarios |
| **Fin de Capa**       | Después de procesar cada capa         | Notificación de progreso, pausa             |
| **Inicio de Pieza**   | Antes de procesar cada pieza          | Numeración de partes, marcas de alineación  |
| **Fin de Pieza**      | Después de procesar cada pieza        | Enfriamiento, pausa de inspección           |

### Creando un Hook

1. **Abrir Configuración de Máquina:**
   - Navega a **Configuración Máquina Hooks**

2. **Seleccionar un disparador:**
   - Elige el evento cuando este hook debe ejecutarse

3. **Escribe tu código G:**
   - El código del hook se inyecta en el punto del disparador
   - Usa variables para valores dinámicos (ver más abajo)

4. **Habilitar/deshabilitar:**
   - Activa/desactiva hooks sin eliminarlos

### Ejemplos de Hooks

#### Inicio de Trabajo: Inicializar Máquina

**Disparador:** Inicio de Trabajo
**Código:**

```gcode
G21         ; Milímetros
G90         ; Posicionamiento absoluto
$H          ; Llevar la máquina al origen
G0 X0 Y0    ; Mover al origen
M3 S0       ; Láser encendido pero potencia 0 (algunos controladores necesitan esto)
M8          ; Asistencia de aire ON
```

**Propósito:** Asegura que la máquina esté en un estado conocido antes de cada trabajo.

---

#### Fin de Trabajo: Volver al Origen y Pitido

**Disparador:** Fin de Trabajo
**Código:**

```gcode
M5          ; Láser OFF
M9          ; Asistencia de aire OFF
G0 X0 Y0    ; Volver al origen
M300 S800 P200  ; Pitido (si es compatible)
```

**Propósito:** Termina el trabajo de manera segura y señala la finalización.

---

#### Inicio de Capa: Añadir Comentario

**Disparador:** Inicio de Capa
**Código:**

```gcode
; Iniciando capa: {layer_name}
; Índice de capa: {layer_index}
```

**Propósito:** Hace el código G más legible para depuración.

---

#### Inicio de Pieza: Numeración de Partes

**Disparador:** Inicio de Pieza
**Código:**

```gcode
; Parte: {workpiece_name}
; Parte {workpiece_index} de {total_workpieces}
```

**Propósito:** Rastrea el progreso en trabajos de múltiples partes.

---

### Orden de Ejecución de Hooks

Para un trabajo con 2 capas, cada una con 2 piezas:

```
[Hook Inicio de Trabajo]
  [Hook Inicio de Capa] (Capa 1)
    [Hook Inicio de Pieza] (Pieza 1)
      ... código G de pieza 1 ...
    [Hook Fin de Pieza] (Pieza 1)
    [Hook Inicio de Pieza] (Pieza 2)
      ... código G de pieza 2 ...
    [Hook Fin de Pieza] (Pieza 2)
  [Hook Fin de Capa] (Capa 1)
  [Hook Inicio de Capa] (Capa 2)
    [Hook Inicio de Pieza] (Pieza 3)
      ... código G de pieza 3 ...
    [Hook Fin de Pieza] (Pieza 3)
    [Hook Inicio de Pieza] (Pieza 4)
      ... código G de pieza 4 ...
    [Hook Fin de Pieza] (Pieza 4)
  [Hook Fin de Capa] (Capa 2)
[Hook Fin de Trabajo]
```

---

## Sustitución de Variables

Tanto las macros como los hooks soportan **sustitución de variables** para inyectar valores dinámicos.

### Variables Disponibles

Las variables usan la sintaxis `{variable_name}` y se reemplazan durante la generación del código G.

**Variables a nivel de trabajo:**

| Variable      | Descripción                       | Valor de Ejemplo |
| ------------- | --------------------------------- | ---------------- |
| `{job_name}`  | Nombre del trabajo/documento actual | "test-job"       |
| `{date}`      | Fecha actual                      | "2025-10-03"     |
| `{time}`      | Hora actual                       | "14:30:25"       |

**Variables a nivel de capa:**

| Variable          | Descripción                         | Valor de Ejemplo |
| ----------------- | ----------------------------------- | ---------------- |
| `{layer_name}`    | Nombre de la capa actual            | "Capa de Corte"  |
| `{layer_index}`   | Índice basado en cero de la capa actual | 0, 1, 2...       |
| `{total_layers}`  | Número total de capas en el trabajo | 3                |

**Variables a nivel de pieza:**

| Variable              | Descripción                            | Valor de Ejemplo |
| --------------------- | -------------------------------------- | ---------------- |
| `{workpiece_name}`    | Nombre de la pieza                     | "Círculo 1"      |
| `{workpiece_index}`   | Índice basado en cero de la pieza actual | 0, 1, 2...       |
| `{total_workpieces}`  | Número total de piezas                 | 5                |

**Variables de máquina:**

| Variable          | Descripción                      | Valor de Ejemplo |
| ----------------- | -------------------------------- | ---------------- |
| `{machine_name}`  | Nombre del perfil de máquina     | "Mi K40"         |
| `{max_speed}`     | Velocidad máxima de corte (mm/min) | 1000             |
| `{work_width}`    | Ancho del área de trabajo (mm)   | 300              |
| `{work_height}`   | Alto del área de trabajo (mm)    | 200              |

### Ejemplo: Notificación de Progreso

**Hook:** Inicio de Capa
**Código:**

```gcode
; ========================================
; Capa {layer_index} de {total_layers}: {layer_name}
; Trabajo: {job_name}
; Hora: {time}
; ========================================
```

**Resultado en código G:**

```gcode
; ========================================
; Capa 0 de 3: Capa de Corte
; Trabajo: test-project
; Hora: 14:30:25
; ========================================
```

---

## Casos de Uso Avanzados

### Configuración Multi-Herramienta

Para máquinas con múltiples láseres o herramientas:

**Hook:** Inicio de Pieza
**Código:**

```gcode
; Seleccionar herramienta para pieza {workpiece_name}
T{tool_number}  ; Comando de cambio de herramienta (si es compatible)
G4 P1           ; Esperar cambio de herramienta
```

### Pausas Condicionales

Añade pausas opcionales para inspección:

**Hook:** Fin de Capa
**Código:**

```gcode
; M0  ; Descomentar para pausar después de cada capa para inspección
```

### Asistencia de Aire por Capa

Controla la asistencia de aire por capa:

**Hook:** Inicio de Capa (para capas de corte)
**Código:**

```gcode
M8  ; Asistencia de aire ON
```

**Hook:** Inicio de Capa (para capas de grabado)
**Código:**

```gcode
M9  ; Asistencia de aire OFF (evita dispersión de polvo para grabado)
```

:::note Hooks Específicos por Capa
Rayforge actualmente no soporta personalización de hooks por capa. Para lograr esto, usa código G condicional o perfiles de máquina separados.
:::

---

## Consideraciones de Seguridad

:::danger Prueba Antes de Producción
Siempre prueba macros y hooks en **modo simulación** o con el láser **deshabilitado** antes de ejecutar en trabajos reales. El código G configurado incorrectamente puede:

- Hacer que la máquina choque contra los límites
- Disparar el láser inesperadamente
- Dañar materiales o equipos
  :::

**Lista de verificación de seguridad:**

- [ ] Las macros incluyen límites de velocidad de avance (`F` parámetro)
- [ ] Las macros verifican los límites de posición
- [ ] Los hooks de Inicio de Trabajo incluyen `M5` o comando de láser apagado
- [ ] Los hooks de Fin de Trabajo apagan el láser (`M5`) y asistencia de aire (`M9`)
- [ ] No hay comandos destructivos sin confirmación
- [ ] Probado en simulación o con láser deshabilitado

---

## Páginas Relacionadas

- [Ajustes de Dispositivo](device) - Referencia de comandos GRBL
- [Dialectos de Código G](../reference/gcode-dialects) - Compatibilidad de código G
- [Ajustes Generales](general) - Configuración de máquina
- [Flujo de Trabajo Multi-Capa](../features/multi-layer) - Usando hooks con capas
