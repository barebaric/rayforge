# Compatibilidad de Firmware

Esta página documenta la compatibilidad de firmware para controladores láser usados con Rayforge.

## Resumen

Rayforge está diseñado principalmente para **controladores basados en GRBL** pero tiene soporte experimental para otros tipos de firmware.

### Matriz de Compatibilidad

| Firmware          | Versión | Estado           | Controlador            | Notas                   |
| ----------------- | ------- | ---------------- | ---------------------- | ----------------------- |
| **GRBL**          | 1.1+    | Completamente soportado | GRBL Serial            | Recomendado             |
| **grblHAL**       | 2023+   | Compatible       | GRBL Serial            | Fork moderno de GRBL    |
| **GRBL**          | 0.9     | Limitado         | GRBL Serial            | Antiguo, puede tener problemas |
| **Smoothieware**  | Todos   | Experimental     | Ninguno (usar controlador GRBL) | Sin probar           |
| **Marlin**        | 2.0+    | Experimental     | Ninguno (usar controlador GRBL) | Requiere modo láser |
| **Otros**         | -       | No soportado     | -                      | Solicitar soporte       |

---

## Firmware GRBL

**Estado:** ✓ Completamente Soportado
**Versiones:** 1.1+
**Controlador:** GRBL Serial

### GRBL 1.1 (Recomendado)

**¿Qué es GRBL 1.1?**

GRBL 1.1 es el firmware más común para máquinas CNC y láser de aficionado. Lanzado en 2017, es estable, bien documentado y ampliamente soportado.

**Funciones soportadas por Rayforge:**

- Comunicación serie (USB)
- Reportes de estado en tiempo real
- Modo láser (M4 potencia constante)
- Lectura/escritura de ajustes ($$, $X=valor)
- Ciclos de homing ($H)
- Sistemas de coordenadas de trabajo (G54)
- Comandos de desplazamiento ($J=)
- Anulación de velocidad de avance
- Límites suaves
- Límites físicos (finales de carrera)

**Limitaciones conocidas:**

- Rango de potencia: 0-1000 (parámetro S)
- Sin conectividad de red (solo USB)
- Memoria a bordo limitada (búfer de código G pequeño)

### Verificando Versión de GRBL

**Consultar versión:**

Conéctate a tu controlador y envía:

```
$I
```

**Ejemplos de respuesta:**

```
[VER:1.1h.20190825:]
[OPT:V,15,128]
```

- `1.1h` = versión de GRBL 1.1h
- La fecha indica compilación

### GRBL 0.9 (Antiguo)

**Estado:** Soporte Limitado

GRBL 0.9 es una versión antigua con algunos problemas de compatibilidad:

**Diferencias:**

- Formato de reporte de estado diferente
- Sin modo láser (M4) - solo usa M3
- Menos ajustes
- Sintaxis de desplazamiento diferente

**Si tienes GRBL 0.9:**

1. **Actualiza a GRBL 1.1** si es posible (recomendado)
2. **Usa M3 en lugar de M4** (potencia menos predecible)
3. **Prueba exhaustivamente** - algunas funciones pueden no funcionar

**Instrucciones de actualización:** Ver [GRBL Wiki](https://github.com/gnea/grbl/wiki)

---

## grblHAL

**Estado:** Compatible
**Versiones:** 2023+
**Controlador:** GRBL Serial

### ¿Qué es grblHAL?

grblHAL es un fork moderno de GRBL con funciones mejoradas:

- Soporte para múltiples hardware de controlador (STM32, ESP32, etc.)
- Red Ethernet/WiFi
- Soporte de tarjeta SD
- Más pines de E/S
- Soporte láser mejorado

**Compatibilidad con Rayforge:**

- **Completamente compatible** - grblHAL mantiene el protocolo GRBL 1.1
- Todas las funciones de GRBL funcionan
- Funciones adicionales (red, SD) aún no soportadas por Rayforge
- Reportes de estado idénticos a GRBL

**Usando grblHAL:**

1. Selecciona el controlador "GRBL Serial" en Rayforge
2. Conecta vía serie USB (igual que GRBL)
3. Todas las funciones funcionan como está documentado para GRBL

**Futuro:** Rayforge puede añadir soporte para funciones específicas de grblHAL (red, etc.)

---

## Smoothieware

**Versiones:** Todos
**Controlador:** GRBL Serial (modo compatibilidad)

### Notas de Compatibilidad

Smoothieware usa sintaxis de código G diferente:

**Diferencias clave:**

| Función          | GRBL           | Smoothieware     |
| ---------------- | -------------- | ---------------- |
| **Láser Encendido** | `M4 S<valor>`  | `M3 S<valor>`    |
| **Rango de Potencia** | 0-1000         | 0.0-1.0 (flotante) |
| **Estado**       | formato `<...>` | Formato diferente |

**Usando Smoothieware con Rayforge:**

1. **Selecciona el dialecto Smoothieware** en ajustes de máquina > Código G > Dialecto
2. **Prueba con potencia baja** primero
3. **Verifica el rango de potencia** coincide con tu configuración
4. **Sin estado en tiempo real** - retroalimentación limitada

**Limitaciones:**

- Reporte de estado no completamente compatible
- La escala de potencia puede diferir
- Comandos de ajustes ($$) no soportados
- Sin probar en hardware real

**Recomendación:** Si es posible, usa firmware compatible con GRBL en su lugar.

---

## Marlin

**Versiones:** 2.0+ con soporte láser
**Controlador:** GRBL Serial

### Marlin para Láser

Marlin 2.0+ puede controlar láseres cuando está configurado apropiadamente.

**Requisitos:**

1. **Firmware Marlin 2.0 o posterior**
2. **Funciones láser habilitadas:**
   ```cpp
   #define LASER_FEATURE
   #define LASER_POWER_INLINE
   ```
3. **Rango de potencia correcto** configurado:
   ```cpp
   #define SPEED_POWER_MAX 1000
   ```

**Compatibilidad:**

- Modo láser M4 soportado
- Código G básico (G0, G1, G2, G3)
- El reporte de estado difiere
- Los comandos de ajustes son diferentes
- Asistencia de aire (M8/M9) puede no funcionar

**Usando Marlin con Rayforge:**

1. **Selecciona el dialecto Marlin** en ajustes de máquina > Código G > Dialecto
2. **Configura Marlin** para uso láser
3. **Prueba el rango de potencia** coincide (0-1000 o 0-255)
4. **Pruebas limitadas** - usa con precaución

**Mejor alternativa:** Usa firmware GRBL en máquinas láser.

---

## Guía de Actualización de Firmware

### Actualizando a GRBL 1.1

**¿Por qué actualizar?**

- Modo láser (M4) para potencia constante
- Mejor reporte de estado
- Más confiable
- Mejor soporte de Rayforge

**Cómo actualizar:**

1. **Identifica tu placa controladora:**
   - Arduino Nano/Uno (ATmega328P)
   - Arduino Mega (ATmega2560)
   - Placa personalizada

2. **Descarga GRBL 1.1:**
   - [GRBL Releases](https://github.com/gnea/grbl/releases)
   - Obtén la última versión 1.1 (1.1h recomendado)

3. **Flashea el firmware:**

   **Usando Arduino IDE:**

   ```
   1. Instala Arduino IDE
   2. Abre el sketch GRBL (grbl.ino)
   3. Selecciona la placa y puerto correctos
   4. Subir
   ```

   **Usando avrdude:**

   ```bash
   avrdude -c arduino -p m328p -P /dev/ttyUSB0 \
           -U flash:w:grbl.hex:i
   ```

4. **Configura GRBL:**
   - Conecta vía serie
   - Envía `$$` para ver ajustes
   - Configura para tu máquina

### Respaldo Antes de Actualizar

**Guarda tus ajustes:**

1. Conéctate al controlador
2. Envía el comando `$$`
3. Copia toda la salida de ajustes
4. Guárdala en un archivo

**Después de actualizar:**

- Restaura los ajustes uno por uno: `$0=10`, `$1=25`, etc.
- O usa los valores predeterminados y reconfigura

---

## Hardware de Controlador

### Controladores Comunes

| Placa                   | Firmware Típico | Soporte Rayforge |
| ----------------------- | --------------- | ---------------- |
| **Arduino CNC Shield**  | GRBL 1.1        | Excelente        |
| **MKS DLC32**           | grblHAL         | Excelente        |
| **Cohesion3D**          | Smoothieware    | Limitado         |
| **Placas SKR**          | Marlin/grblHAL  | Variable         |
| **Ruida**               | Propietario     | No soportado     |
| **Trocen**              | Propietario     | No soportado     |
| **TopWisdom**           | Propietario     | No soportado     |

### Controladores Recomendados

Para mejor compatibilidad con Rayforge:

1. **Arduino Nano + CNC Shield** (GRBL 1.1)
   - Económico (~$10-20)
   - Fácil de flashear
   - Bien documentado

2. **MKS DLC32** (grblHAL)
   - Moderno (basado en ESP32)
   - Capaz de WiFi
   - Desarrollo activo

3. **Placas GRBL personalizadas**
   - Muchas disponibles en marketplaces
   - Verifica soporte de GRBL 1.1+

---

## Configuración de Firmware

### Ajustes de GRBL para Láser

**Ajustes esenciales:**

```
$30=1000    ; Potencia máxima de husillo/láser (1000 = 100%)
$31=0       ; Potencia mínima de husillo/láser
$32=1       ; Modo láser habilitado (1 = on)
```

**Ajustes de máquina:**

```
$100=80     ; Pasos/mm X (calibra para tu máquina)
$101=80     ; Pasos/mm Y
$110=3000   ; Tasa máxima X (mm/min)
$111=3000   ; Tasa máxima Y
$120=100    ; Aceleración X (mm/seg)
$121=100    ; Aceleración Y
$130=300    ; Recorrido máximo X (mm)
$131=200    ; Recorrido máximo Y (mm)
```

**Ajustes de seguridad:**

```
$20=1       ; Límites suaves habilitados
$21=1       ; Límites físicos habilitados (si tienes finales de carrera)
$22=1       ; Homing habilitado
```

### Probando el Firmware

**Secuencia de prueba básica:**

1. **Prueba de conexión:**

   ```
   Enviar: ?
   Esperar: <Idle|...>
   ```

2. **Verificación de versión:**

   ```
   Enviar: $I
   Esperar: [VER:1.1...]
   ```

3. **Verificación de ajustes:**

   ```
   Enviar: $$
   Esperar: $0=..., $1=..., etc.
   ```

4. **Prueba de movimiento:**

   ```
   Enviar: G91 G0 X10
   Esperar: La máquina se mueve 10mm en X
   ```

5. **Prueba de láser (potencia muy baja):**
   ```
   Enviar: M4 S10
   Esperar: El láser se enciende (tenue)
   Enviar: M5
   Esperar: El láser se apaga
   ```

---

## Solución de Problemas de Firmware

### El Firmware No Responde

**Síntomas:**

- Sin respuesta a comandos
- La conexión falla
- El estado no se reporta

**Diagnóstico:**

1. **Revisa la velocidad de transmisión:**
   - GRBL 1.1 por defecto: 115200
   - GRBL 0.9: 9600
   - Prueba ambos

2. **Revisa el cable USB:**
   - Cable de datos, no solo de carga
   - Reemplaza con un cable conocido como bueno

3. **Revisa el puerto:**
   - Linux: `/dev/ttyUSB0` o `/dev/ttyACM0`
   - Windows: COM3, COM4, etc.
   - Puerto correcto seleccionado en Rayforge

4. **Prueba con terminal:**
   - Usa screen, minicom, o PuTTY
   - Envía `?` y ve si obtienes respuesta

### El Firmware Falla

**Síntomas:**

- El controlador se bloquea durante el trabajo
- Desconexiones aleatorias
- Comportamiento inconsistente

**Posibles causas:**

1. **Desbordamiento de búfer** - Archivo de código G demasiado complejo
2. **Ruido eléctrico** - Mala conexión a tierra o EMI
3. **Bug de firmware** - Actualiza a la última versión
4. **Problema de hardware** - Controlador defectuoso

**Soluciones:**

- Actualiza el firmware a la última versión estable
- Simplifica el código G (reduce precisión, menos segmentos)
- Añade cuentas de ferrita al cable USB
- Mejora la conexión a tierra y el enrutamiento de cables

### Firmware Incorrecto

**Síntomas:**

- Comandos rechazados
- Comportamiento inesperado
- Mensajes de error

**Solución:**

1. Consulta la versión del firmware: `$I`
2. Compara con las expectativas de Rayforge
3. Actualiza o selecciona el dialecto correcto

---

## Soporte de Firmware Futuro

### Funciones Solicitadas

Los usuarios han solicitado soporte para:

- **Controladores Ruida** - Controladores láser chinos
- **Trocen/AWC** - Controladores láser comerciales
- **ESP32 WiFi** - Conectividad de red para grblHAL
- **API de Láser** - API de máquina directa (sin código G)

**Estado:** Actualmente no soportados. Las solicitudes de funciones son bienvenidas en GitHub.

### Contribuyendo

Para añadir soporte de firmware:

1. Implementa el controlador en `rayforge/machine/driver/`
2. Define el dialecto de código G en `rayforge/machine/models/dialect.py`
3. Prueba exhaustivamente en hardware real
4. Envía pull request con documentación

---

## Páginas Relacionadas

- [Dialectos de Código G](gcode-dialects) - Detalles de dialectos
- [Ajustes de Dispositivo](../machine/device) - Configuración de GRBL
- [Problemas de Conexión](../troubleshooting/connection) - Solución de problemas de conexión
- [Ajustes Generales](../machine/general) - Configuración de máquina
