# Soporte de Dialectos de Código G

Rayforge soporta múltiples dialectos de código G para trabajar con diferentes
firmware de controlador.

## Selección Automática de Dialecto

Cuando se descubre un dispositivo GRBL (a través del [asistente de configuración](../getting-started/first-time-setup.md)
o el descubrimiento de dispositivos), Rayforge lee las flags de compilación `$I`
del firmware y selecciona automáticamente el dialecto de código G que mejor
coincide — generalmente no necesitas elegir uno manualmente. Aún puedes
sobreescribir la elección después en los ajustes de dialecto de código G de la
máquina.

## Dialectos Soportados

Rayforge actualmente soporta estos dialectos de código G:

| Dialecto                                      | Firmware     | Uso Común                               |
| --------------------------------------------- | ------------ | --------------------------------------- |
| **Grbl (Compat)**                             | GRBL 1.1+    | Láseres de diodo, CNC de aficionado     |
| **Grbl (Compat, sin eje Z)**                  | GRBL 1.1+    | Cortadores láser 2D sin Z               |
| **Grbl Raster**                               | GRBL 1.1+    | Optimizado para trabajo raster          |
| **GRBL Dinámico (Consciente de Profundidad)** | GRBL 1.1+    | Grabado láser consciente de profundidad |
| **GRBL Dinámico (sin eje Z)**                 | GRBL 1.1+    | Grabado láser consciente de profundidad |
| **LinuxCNC**                                  | LinuxCNC     | Soporte nativo de Bézier (G5)           |
| **Mach4 (M67 Analog)**                        | Mach4        | Grabado ráster de alta velocidad        |
| **Smoothieware**                              | Smoothieware | Cortadores láser, CNC                   |
| **Marlin**                                    | Marlin 2.0+  | Impresoras 3D con láser                 |

:::note Dialectos Recomendados
:::

**Grbl (Compat)** es el dialecto más probado y recomendado para aplicaciones
láser estándar.

**Grbl Raster** está optimizado para grabado raster en controladores GRBL. Mantiene
el láser en modo de potencia dinámica (M4) continuamente y omite comandos de
velocidad de avance redundantes, resultando en una salida de código G más suave
y compacta.

**GRBL Dinámico (Consciente de Profundidad)** es recomendado para grabado láser
consciente de profundidad donde la potencia varía durante los cortes (ej.,
grabado de profundidad variable).

**LinuxCNC** admite curvas Bézier cúbicas nativas a través del comando G5, lo
que produce un código G muy suave y compacto para caminos curvos. Cuando uses
este dialecto, activa la opción «Soportar curvas Bézier» en Ajustes Avanzados
de Máquina para aprovechar la salida G5.

---

## Mach4 (M67 Analog)

El dialecto **Mach4 (M67 Analog)** está diseñado para grabado ráster de alta
velocidad con controladores Mach4. Utiliza el comando M67 con salida analógica
para un control preciso de la potencia del láser.

### Características Principales

- **Salida Analógica M67**: Utiliza `M67 E0 Q<0-255>` para la potencia del
  láser en lugar de comandos S en línea
- **Presión de Búfer Reducida**: Al separar los comandos de potencia de los
  comandos de movimiento, el búfer del controlador sufre menos estrés durante
  operaciones de alta velocidad
- **Ráster de Alta Velocidad**: Optimizado para operaciones de grabado ráster
  rápidas

### Cuándo Usar

Usa este dialecto cuando:

- Tengas un controlador Mach4 con capacidad de salida analógica
- Necesites grabado ráster de alta velocidad
- Tu controlador experimente desbordamiento de búfer con comandos S en línea
  estándar

### Formato de Comando

El dialecto genera código G como:

```gcode
M67 E0 Q127  ; Establecer potencia del láser al 50% (127/255)
G1 X100 Y200 F1000  ; Mover a posición
M67 E0 Q0    ; Apagar láser
```

---

## Creando un Dialecto Personalizado

Para crear un dialecto de código G personalizado basado en un dialecto integrado:

1. Abre **Ajustes de Máquina** → **Dialecto de Código G**
2. Haz clic en el icono **Copiar** en un dialecto integrado para crear un nuevo
   dialecto personalizado
3. Edita los ajustes del dialecto según sea necesario
4. Guarda tu dialecto personalizado

Cada dialecto personalizado es una copia independiente. Cambiar un dialecto
nunca afecta a otros, por lo que puedes experimentar libremente sin preocuparte
por dañar una configuración existente. Los dialectos personalizados se almacenan
en tu directorio de configuración y pueden compartirse.

### Ajustes del Dialecto

Al editar un dialecto personalizado, la página de Ajustes ofrece estas opciones:

**Modo Láser Continuo** mantiene el láser en modo de potencia dinámica (M4) activo
durante todo el trabajo en lugar de alternar M4/M5 entre segmentos. Esto es útil
para grabado raster donde el láser necesita permanecer encendido continuamente
durante las líneas de escaneo.

**Velocidad de Avance Modal** omite el parámetro de velocidad de avance (F) de los
comandos de movimiento cuando no ha cambiado desde el último comando. Esto produce
código G más compacto y reduce la cantidad de datos enviados al controlador.

### Comando Separado de Encendido del Láser para Enfoque

Algunos dialectos soportan la configuración de un comando separado para encender
el láser a baja potencia, lo cual es útil para el modo de enfoque. Esto te
permite usar un comando diferente para el comportamiento visual de «puntero
láser» que el utilizado durante el corte o grabado real. Revisa la página de
ajustes de tu dialecto para esta opción.

---

## Espacios Reservados de Plantillas

Al crear o editar un dialecto personalizado, cada plantilla de comando utiliza
[cadenas de formato de Python](https://docs.python.org/3/library/string.html#format-string-syntax)
con espacios reservados para inyectar valores dinámicos. Usa la sintaxis
`{nombre}` o `{nombre:.0f}` (ej., `{power:.0f}` para formatear como entero sin
decimales).

### Espacios Reservados Disponibles por Plantilla

| Plantilla           | Espacios Reservados                                                                                          |
| ------------------- | ------------------------------------------------------------------------------------------------------------ |
| **Laser Encendido** | `power`                                                                                                      |
| **Laser Enfoque**   | `power`                                                                                                      |
| **Laser Apagado**   | _(ninguno)_                                                                                                  |
| **Cambio Herram.**  | `tool_number`                                                                                                |
| **Ajustar Veloc.**  | `speed`                                                                                                      |
| **Mov. Rápido**     | `x`, `y`, `z`, `x_cmd`, `y_cmd`, `z_cmd`, `extra_cmd`, `f_command`, `s_command`                              |
| **Mov. Lineal**     | `x`, `y`, `z`, `x_cmd`, `y_cmd`, `z_cmd`, `extra_cmd`, `f_command`, `s_command`, `i`, `j`, `power`           |
| **Arco (CW)**       | `x`, `y`, `z`, `x_cmd`, `y_cmd`, `z_cmd`, `extra_cmd`, `f_command`, `s_command`, `i`, `j`, `power`           |
| **Arco (CCW)**      | `x`, `y`, `z`, `x_cmd`, `y_cmd`, `z_cmd`, `extra_cmd`, `f_command`, `s_command`, `i`, `j`, `power`           |
| **Bézier Cúbico**   | `x`, `y`, `z`, `x_cmd`, `y_cmd`, `z_cmd`, `extra_cmd`, `f_command`, `s_command`, `i`, `j`, `p`, `q`, `power` |
| **Aire On/Off**     | _(ninguno)_                                                                                                  |
| **Origen Todos**    | _(ninguno)_                                                                                                  |
| **Origen Eje**      | `axis_letter`                                                                                                |
| **Mover A**         | `speed`, `x`, `y`, `z`                                                                                       |
| **Jog**             | `speed`                                                                                                      |
| **Limpiar Alarma**  | _(ninguno)_                                                                                                  |
| **Ajuste WCS**      | `p_num`, `x`, `y`, `z`                                                                                       |
| **Ciclo Sonda**     | `axis_letter`, `max_travel`, `feed_rate`                                                                     |
| **Espera**          | `seconds`, `milliseconds`                                                                                    |

### Referencia de Espacios Reservados

#### Coordenadas

| Espacio Reservado | Descripción                                                                                                               |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `x`               | Coordenada X objetivo como float (ej., `100.0`)                                                                           |
| `y`               | Coordenada Y objetivo como float (ej., `200.0`)                                                                           |
| `z`               | Coordenada Z objetivo como float (ej., `5.0`)                                                                             |
| `x_cmd`           | Cadena de comando del eje X, ej., `" X100.0"`. Se omite si no cambia (si "Omitir coordenadas no cambiadas" está activado) |
| `y_cmd`           | Cadena de comando del eje Y, ej., `" Y200.0"`. Se omite si no cambia                                                      |
| `z_cmd`           | Cadena de comando del eje Z, ej., `" Z5.0"`. Se omite si no cambia                                                        |
| `extra_cmd`       | Cadena de comando para ejes extra (A, B, C), ej., `" A90.0"`. Vacía si no hay ejes extra configurados                     |

#### Movimiento

| Espacio Reservado | Descripción                                                                                                    |
| ----------------- | -------------------------------------------------------------------------------------------------------------- |
| `f_command`       | Cadena de comando de velocidad de avance, ej., `" F3000"`. Se omite si es modal y no cambia                    |
| `s_command`       | Cadena de comando de husillo/potencia, ej., `" S500"`. Usada en modos dinámico/raster y en modo láser continuo |
| `i`               | Desplazamiento X del punto de control del arco o Bézier desde la posición inicial                              |
| `j`               | Desplazamiento Y del punto de control del arco o Bézier desde la posición inicial                              |
| `p`               | Desplazamiento X del segundo punto de control Bézier desde la posición final (solo Bézier Cúbico)              |
| `q`               | Desplazamiento Y del segundo punto de control Bézier desde la posición final (solo Bézier Cúbico)              |

#### Potencia y velocidad

| Espacio Reservado | Descripción                                                                                       |
| ----------------- | ------------------------------------------------------------------------------------------------- |
| `power`           | Valor absoluto de potencia del láser como float. Soporta formato, ej., `{power:.0f}` para enteros |
| `speed`           | Valor de velocidad (para comandos Mover A y Jog)                                                  |
| `tool_number`     | Número de herramienta/cabeza láser                                                                |

#### Máquina y Sondaje

| Espacio Reservado | Descripción                                                            |
| ----------------- | ---------------------------------------------------------------------- |
| `axis_letter`     | Letra de eje única, ej., `"X"`, `"Y"`, `"Z"` (para Origen Eje y Sonda) |
| `p_num`           | Número P del WCS (ej., `1` para G54)                                   |
| `max_travel`      | Distancia máxima de viaje de la sonda (solo Ciclo Sonda)               |
| `feed_rate`       | Velocidad de avance de la sonda (solo Ciclo Sonda)                     |

#### Espera

| Espacio Reservado | Descripción                                                  |
| ----------------- | ------------------------------------------------------------ |
| `seconds`         | Duración de espera en segundos como float (ej., `1.5`)       |
| `milliseconds`    | Duración de espera en milisegundos como entero (ej., `1500`) |

### Consejos

- Se admiten **especificaciones de formato**: `{power:.0f}` formatea la potencia como entero,
  `{power:.2f}` con dos decimales.
- La configuración **«Omitir coordenadas no cambiadas»** controla si `x_cmd`, `y_cmd`
  y `z_cmd` se dejan vacíos cuando la posición del eje no ha cambiado desde el
  último comando. Esto reduce el tamaño del código G.
- La configuración **«Velocidad de Avance Modal»** controla si `f_command` se omite
  cuando la velocidad de avance no ha cambiado.
- Deja un campo de plantilla **vacío** para omitir ese comando por completo
  (ej., establecer `bezier_cubic` en `""` desactiva la salida Bézier nativa
  y usa linealización como alternativa).

---

## Páginas Relacionadas

- [Exportando Código G](../files/exporting.md) - Ajustes de exportación
- [Compatibilidad de Firmware](firmware) - Versiones de firmware
- [Ajustes de Dispositivo](../machine/device.md) - Configuración de GRBL
- [Macros y Hooks](../machine/hooks-macros.md) - Inyección de código G personalizado
