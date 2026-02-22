# Ajustes de Dispositivo

La página de Dispositivo en Configuración de Máquina te permite leer y aplicar ajustes directamente a tu dispositivo conectado (controlador). Estos también se conocen como ajustes "dólar" o ajustes `$$` en GRBL.

![Ajustes de Dispositivo](/screenshots/machine-device.png)

:::warning Precaución al Cambiar Ajustes
Los ajustes de firmware incorrectos pueden hacer que tu máquina se comporte de manera impredecible, pierda posición o incluso dañe el hardware. Siempre registra los valores originales antes de hacer cambios, y modifica un ajuste a la vez.
:::

## Resumen

La página de Dispositivo proporciona acceso directo a los ajustes de firmware de tu controlador. Aquí es donde puedes:

- Leer ajustes actuales del dispositivo
- Modificar ajustes individuales
- Aplicar cambios al dispositivo

Los ajustes de firmware controlan:

- **Parámetros de movimiento**: Límites de velocidad, aceleración, calibración
- **Interruptores de límite**: Comportamiento de homing, límites suaves/físicos
- **Control de láser**: Rango de potencia, habilitación del modo láser
- **Configuración eléctrica**: Inversiones de pines, pullups
- **Reportes**: Formato y frecuencia de mensajes de estado

Estos ajustes se almacenan en tu controlador (no en Rayforge) y persisten a través de ciclos de energía.

## Leyendo Ajustes

Haz clic en el botón **Leer del Dispositivo** para obtener los ajustes actuales de tu controlador conectado. Esto requiere:

- Que la máquina esté conectada
- Que el controlador soporte la lectura de ajustes del dispositivo

## Aplicando Ajustes

Después de modificar los ajustes, los cambios se aplican al dispositivo. El dispositivo puede:

- Reiniciarse temporalmente
- Desconectarse y reconectarse
- Requerir un ciclo de energía para algunos cambios

## Acceso a la Consola

También puedes ver/modificar ajustes vía la consola de código G:

**Ver todos los ajustes:**
```
$$
```

**Ver un solo ajuste:**
```
$100
```

**Modificar ajuste:**
```
$100=80.0
```

**Restaurar valores predeterminados:**
```
$RST=$
```

:::danger Restaurar Valores Predeterminados Borra Todos los Ajustes
El comando `$RST=$` reinicia todos los ajustes de GRBL a los valores de fábrica. ¡Perderás cualquier calibración y ajuste! ¡Haz una copia de seguridad de tus ajustes primero!
:::

---

## Ajustes Críticos para Láseres

Estos ajustes son los más importantes para la operación láser:

### $32 - Modo Láser

**Valor:** 0 = Deshabilitado, 1 = Habilitado

**Propósito:** Habilita funciones específicas de láser en GRBL

**Cuando está habilitado (1):**
- El láser se apaga automáticamente durante movimientos G0 (rápidos)
- La potencia se ajusta dinámicamente durante la aceleración/desaceleración
- Previene quemaduras accidentales durante el posicionamiento

**Cuando está deshabilitado (0):**
- El láser se comporta como un husillo (modo CNC)
- No se apaga durante los rápidos
- **¡Peligroso para uso láser!**

:::warning Siempre Habilita el Modo Láser
$32 debería **siempre** establecerse en 1 para cortadores láser. El modo láser deshabilitado puede causar quemaduras no intencionales y riesgos de incendio.
:::

### $30 y $31 - Rango de Potencia del Láser

**$30 - Potencia Máxima del Láser (RPM)**
**$31 - Potencia Mínima del Láser (RPM)**

**Propósito:** Define el rango de potencia para comandos S

**Valores típicos:**
- $30=1000, $31=0 (rango S0-S1000, más común)
- $30=255, $31=0 (rango S0-S255, algunos controladores)

:::tip Coincidir con la Configuración de Rayforge
El ajuste "Potencia Máxima" en tus [Ajustes de Láser](laser) debería coincidir con tu valor $30. Si $30=1000, establece la potencia máxima en 1000 en Rayforge.
:::

### $130 y $131 - Recorrido Máximo

**$130 - Recorrido Máximo X (mm)**
**$131 - Recorrido Máximo Y (mm)**

**Propósito:** Define el área de trabajo de tu máquina

**Por qué importa:**
- Los límites suaves ($20) usan estos valores para prevenir choques
- Define los límites del sistema de coordenadas
- Debe coincidir con el tamaño físico de tu máquina

---

## Referencia de Ajustes

### Configuración de Motores Paso a Paso ($0-$6)

Controla las señales eléctricas y el tiempo de los motores paso a paso.

| Ajuste | Descripción | Valor Típico |
|--------|-------------|--------------|
| $0 | Tiempo de pulso de paso (μs) | 10 |
| $1 | Retraso de paso inactivo (ms) | 25 |
| $2 | Máscara de inversión de pulso de paso | 0 |
| $3 | Máscara de inversión de dirección de paso | 0 |
| $4 | Invertir pin de habilitación de paso | 0 |
| $5 | Invertir pines de límite | 0 |
| $6 | Invertir pin de sonda | 0 |

### Límites y Homing ($20-$27)

Controla los interruptores de límite y el comportamiento de homing.

| Ajuste | Descripción | Valor Típico |
|--------|-------------|--------------|
| $20 | Habilitar límites suaves | 0 o 1 |
| $21 | Habilitar límites físicos | 0 |
| $22 | Habilitar ciclo de homing | 0 o 1 |
| $23 | Invertir dirección de homing | 0 |
| $24 | Velocidad de localización de homing (mm/min) | 25 |
| $25 | Velocidad de búsqueda de homing (mm/min) | 500 |
| $26 | Retraso de rebote de homing (ms) | 250 |
| $27 | Distancia de retroceso de homing (mm) | 1.0 |

### Husillo y Láser ($30-$32)

| Ajuste | Descripción | Valor Láser |
|--------|-------------|-------------|
| $30 | Velocidad máxima del husillo | 1000.0 |
| $31 | Velocidad mínima del husillo | 0.0 |
| $32 | Habilitar modo láser | 1 |

### Calibración de Ejes ($100-$102)

Define cuántos pasos del motor paso a paso equivalen a un milímetro de movimiento.

| Ajuste | Descripción | Notas |
|--------|-------------|-------|
| $100 | Pasos/mm X | Depende de la relación polea/correa |
| $101 | Pasos/mm Y | Usualmente igual que X |
| $102 | Pasos/mm Z | No se usa en la mayoría de láseres |

**Calculando pasos/mm:**
```
pasos/mm = (pasos_motor_por_rev × micropasos) / (dientes_polea × paso_correa)
```

**Ejemplo:** 200 pasos/rev, 16 micropasos, polea de 20 dientes, correa GT2:
```
pasos/mm = (200 × 16) / (20 × 2) = 3200 / 40 = 80
```

### Velocidad y Aceleración de Ejes ($110-$122)

| Ajuste | Descripción | Valor Típico |
|--------|-------------|--------------|
| $110 | Tasa máxima X (mm/min) | 5000.0 |
| $111 | Tasa máxima Y (mm/min) | 5000.0 |
| $112 | Tasa máxima Z (mm/min) | 500.0 |
| $120 | Aceleración X (mm/seg²) | 500.0 |
| $121 | Aceleración Y (mm/seg²) | 500.0 |
| $122 | Aceleración Z (mm/seg²) | 100.0 |

### Recorrido de Ejes ($130-$132)

| Ajuste | Descripción | Notas |
|--------|-------------|-------|
| $130 | Recorrido máximo X (mm) | Ancho del área de trabajo |
| $131 | Recorrido máximo Y (mm) | Profundidad del área de trabajo |
| $132 | Recorrido máximo Z (mm) | Recorrido Z (si aplica) |

---

## Ejemplo de Configuración Común

### Láser de Diodo Típico (300×400mm)

```gcode
$0=10          ; Pulso de paso 10μs
$1=255         ; Retraso de paso inactivo 255ms
$2=0           ; Sin inversión de paso
$3=0           ; Sin inversión de dirección
$4=0           ; Sin inversión de habilitación
$5=0           ; Sin inversión de límite
$10=1          ; Reportar WPos
$11=0.010      ; Desviación de unión 0.01mm
$12=0.002      ; Tolerancia de arco 0.002mm
$13=0          ; Reportar en mm
$20=1          ; Límites suaves habilitados
$21=0          ; Límites físicos deshabilitados
$22=1          ; Homing habilitado
$23=0          ; Home al mínimo
$24=50.0       ; Velocidad de homing 50mm/min
$25=1000.0     ; Búsqueda de homing 1000mm/min
$26=250        ; Rebote de homing 250ms
$27=2.0        ; Retroceso de homing 2mm
$30=1000.0     ; Potencia máxima S1000
$31=0.0        ; Potencia mínima S0
$32=1          ; Modo láser ON
$100=80.0      ; Pasos/mm X
$101=80.0      ; Pasos/mm Y
$102=80.0      ; Pasos/mm Z
$110=5000.0    ; Tasa máxima X
$111=5000.0    ; Tasa máxima Y
$112=500.0     ; Tasa máxima Z
$120=500.0     ; Aceleración X
$121=500.0     ; Aceleración Y
$122=100.0     ; Aceleración Z
$130=400.0     ; Recorrido máximo X
$131=300.0     ; Recorrido máximo Y
$132=0.0       ; Recorrido máximo Z
```

---

## Haciendo Copia de Seguridad de los Ajustes

### Procedimiento de Copia de Seguridad

1. **Vía Rayforge:**
   - Abre el panel de Ajustes de Dispositivo
   - Haz clic en "Exportar Ajustes"
   - Guarda el archivo como `grbl-backup-YYYY-MM-DD.txt`

2. **Vía consola:**
   - Envía el comando `$$`
   - Copia toda la salida a un archivo de texto
   - Guarda con la fecha

### Procedimiento de Restauración

1. Abre el archivo de respaldo
2. Envía cada línea (`$100=80.0`, etc.) vía consola
3. Verifica con el comando `$$`

:::tip Copias de Seguridad Regulares
Haz copia de seguridad de tus ajustes después de cualquier calibración o ajuste. Almacena las copias de seguridad en un lugar seguro.
:::

---

## Ver También

- [Ajustes Generales](general) - Nombre de máquina y ajustes de velocidad
- [Ajustes de Láser](laser) - Configuración del cabezal láser
- [Problemas de Conexión](../troubleshooting/connection) - Solucionando problemas de conexión

## Recursos Externos

- [GRBL v1.1 Configuration](https://github.com/gnea/grbl/wiki/Grbl-v1.1-Configuration)
- [GRBL v1.1 Commands](https://github.com/gnea/grbl/wiki/Grbl-v1.1-Commands)
- [Grbl_ESP32 Documentation](https://github.com/bdring/Grbl_Esp32/wiki)
