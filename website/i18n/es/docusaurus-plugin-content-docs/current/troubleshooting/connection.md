# Problemas de Conexión

Esta página te ayuda a diagnosticar y resolver problemas al conectar Rayforge a tu máquina láser vía conexión serie.

## Diagnóstico Rápido

### Síntomas

Los problemas comunes de conexión incluyen:

- Error "El puerto debe estar configurado" al intentar conectar
- La conexión falla repetidamente y se reconecta
- El puerto serie no aparece en la lista de puertos
- Errores de "Permiso denegado" al intentar abrir el puerto serie
- El dispositivo parece conectarse pero no responde a comandos

---

## Problemas Comunes y Soluciones

### No Se Detectan Puertos Serie

**Problema:** El menú desplegable de puerto serie está vacío o no muestra tu dispositivo.

**Diagnóstico:**

1. Verifica si tu dispositivo está encendido y conectado vía USB
2. Prueba desconectando y reconectando el cable USB
3. Prueba el cable USB con otro dispositivo (los cables pueden fallar)
4. Prueba un puerto USB diferente en tu computadora

**Soluciones:**

**Linux:**

Primero, añade tu usuario al grupo `dialout` (requerido para instalaciones
Snap y no Snap en la mayoría de distribuciones basadas en Debian):

```bash
sudo usermod -a -G dialout $USER
```

Luego cierra sesión y vuelve a entrar para que el cambio surta efecto.

Si estás usando la versión Snap, también necesitas otorgar permisos de puerto
serie:

```bash
sudo snap connect rayforge:serial-port
```

Ver [Permisos de Snap](snap-permissions) para configuración detallada de Linux.

**Windows:**
1. Abre el Administrador de Dispositivos (Win+X, luego selecciona Administrador de Dispositivos)
2. Busca en "Puertos (COM y LPT)" tu dispositivo
3. Si ves un ícono de advertencia amarillo, actualiza o reinstala el controlador
4. Anota el número de puerto COM (ej., COM3)
5. Si el dispositivo no está listado, el cable USB o el controlador pueden estar defectuosos

**macOS:**
1. Revisa Información del Sistema → USB para verificar que el dispositivo es reconocido
2. Instala controladores CH340/CH341 si tu controlador usa este chipset
3. Busca dispositivos `/dev/tty.usbserial*` o `/dev/cu.usbserial*`

### Errores de Permiso Denegado

**Problema:** Obtienes errores de "Permiso denegado" o similares al intentar conectar.

**En Linux:**

Tu usuario necesita estar en el grupo `dialout` (o `uucp` en algunas
distribuciones). Esto es requerido para **instalaciones Snap y no Snap** en
distribuciones basadas en Debian para evitar mensajes AppArmor DENIED:

```bash
# Añádete al grupo dialout
sudo usermod -a -G dialout $USER

# Verifica que estás en el grupo (después de cerrar sesión/entrar)
groups | grep dialout
```

**Importante:** Debes cerrar sesión y volver a entrar (o reiniciar) para que
los cambios de grupo surtan efecto.

**Adicionalmente para instalaciones Snap**, otorga acceso al puerto serie al
snap:

```bash
sudo snap connect rayforge:serial-port
```

Ver la guía de [Permisos de Snap](snap-permissions) para más detalles.

**En Windows:**

Cierra cualquier otra aplicación que pueda estar usando el puerto serie, incluyendo:
- Instancias previas de Rayforge
- Herramientas de monitor serie
- Otro software láser
- Arduino IDE o herramientas similares

### Puerto Serie Incorrecto Seleccionado

**Problema:** Rayforge conecta pero la máquina no responde.

**Diagnóstico:**

Puede que hayas seleccionado el puerto incorrecto, especialmente si tienes múltiples dispositivos USB conectados.

**Solución:**

1. Desconecta todos los otros dispositivos serie USB
2. Anota qué puertos están disponibles en Rayforge
3. Conecta tu controlador láser
4. Refresca la lista de puertos - el nuevo puerto es tu láser
5. En Linux, los controladores láser típicamente aparecen como:
   - `/dev/ttyUSB0` (común para chipsets CH340)
   - `/dev/ttyACM0` (común para controladores USB nativos)
6. En Windows, anota el puerto COM del Administrador de Dispositivos
7. Evita puertos nombrados `/dev/ttyS*` en Linux - estos son puertos serie de hardware, no USB

:::warning Puertos Serie de Hardware
Rayforge te advertirá si seleccionas puertos `/dev/ttyS*` en Linux, ya que estos típicamente no son dispositivos GRBL basados en USB. Los puertos serie USB usan `/dev/ttyUSB*` o `/dev/ttyACM*`.
:::


### Velocidad de Transmisión Incorrecta

**Problema:** La conexión se establece pero los comandos no funcionan o producen respuestas ilegibles.

**Solución:**

Los controladores GRBL típicamente usan una de estas velocidades de transmisión:

- **115200** (más común, GRBL 1.1+)
- **9600** (versiones de GRBL más antiguas)
- **250000** (menos común, algunos firmware personalizados)

Prueba diferentes velocidades de transmisión en los ajustes de dispositivo de Rayforge. La más común es **115200**.

### La Conexión Se Caé Continuamente

**Problema:** Rayforge conecta exitosamente pero sigue desconectándose y reconectándose.

**Posibles Causas:**

1. **Cable USB defectuoso** - Reemplaza con un cable conocido como bueno (preferiblemente corto, <2m)
2. **Problemas de energía USB** - Prueba un puerto USB diferente, preferiblemente en la computadora misma en lugar de un hub
3. **EMI/Interferencia** - Mantén los cables USB alejados de cables de motores y fuentes de alimentación de alto voltaje
4. **Problemas de firmware** - Actualiza tu firmware GRBL si es posible
5. **Conflictos de puerto USB** - En Windows, prueba diferentes puertos USB

**Pasos para Solución de Problemas:**

```bash
# En Linux, monitorea los registros del sistema mientras conectas:
sudo dmesg -w
```

Busca mensajes como:
- "USB disconnect" - indica problemas físicos/de cable
- "device descriptor read error" - a menudo un problema de energía o cable

### El Dispositivo No Responde Después de Conectar

**Problema:** El estado de conexión muestra "Conectado" pero la máquina no responde a comandos.

**Diagnóstico:**

1. Verifica que el tipo de firmware correcto esté seleccionado (GRBL vs otro)
2. Verifica que la máquina esté encendida (controlador y fuente de alimentación)
3. Revisa si la máquina está en estado de alarma (requiere homing o limpieza de alarma)

**Solución:**

Prueba enviando un comando manual en la Consola:

- `?` - Solicitar reporte de estado
- `$X` - Limpiar alarma
- `$H` - Llevar la máquina al origen

Si no hay respuesta, verifica doblemente la velocidad de transmisión y la selección de puerto.

---

## Mensajes de Estado de Conexión

Rayforge muestra diferentes estados de conexión:

| Estado          | Significado | Acción |
| --------------- | ------------|--------|
| **Desconectado** | No conectado a ningún dispositivo | Configurar puerto y conectar |
| **Conectando**   | Intentando establecer conexión | Esperar, o verificar configuración si se queda atascado |
| **Conectado**    | Conectado exitosamente y recibiendo estado | Listo para usar |
| **Error**        | La conexión falló con un error | Revisar mensaje de error para detalles |
| **Durmiendo**    | Esperando antes de intento de reconexión | Conexión previa fallida, reintentando en 5s |

---

## Probando Tu Conexión

### Prueba de Conexión Paso a Paso

1. **Configurar la máquina:**
   - Abre Configuración → Máquina
   - Selecciona o crea un perfil de máquina
   - Elige el controlador correcto (GRBL Serial)
   - Selecciona el puerto serie
   - Establece la velocidad de transmisión (típicamente 115200)

2. **Intentar conexión:**
   - Haz clic en "Conectar" en el panel de control de la máquina
   - Observa el indicador de estado de conexión

3. **Verificar comunicación:**
   - Si está conectado, prueba enviando una consulta de estado
   - La máquina debería reportar su posición y estado

4. **Probar comandos básicos:**
   - Prueba homing (`$H`) si tu máquina tiene interruptores de límite
   - O limpia alarmas (`$X`) si es necesario

### Usando Registros de Depuración

Rayforge incluye registro detallado para problemas de conexión. Para habilitar el registro de depuración:

```bash
# Ejecutar Rayforge desde terminal con registro de depuración
rayforge --loglevel DEBUG
```

Revisa los registros para:
- Intentos de conexión y fallos
- Datos serie transmitidos (TX) y recibidos (RX)
- Mensajes de error con trazas de pila

---

## Solución de Problemas Avanzada

### Verificando Disponibilidad del Puerto Manualmente

**Linux:**
```bash
# Listar todos los dispositivos serie USB
ls -l /dev/ttyUSB* /dev/ttyACM*

# Verificar permisos
ls -l /dev/ttyUSB0  # Reemplazar con tu puerto

# Debería mostrar: crw-rw---- 1 root dialout
# Necesitas estar en el grupo 'dialout'

# Probar puerto manualmente
sudo minicom -D /dev/ttyUSB0 -b 115200
```

**Windows:**
```powershell
# Listar puertos COM en PowerShell
[System.IO.Ports.SerialPort]::getportnames()

# O usar Administrador de Dispositivos:
# Win + X → Administrador de Dispositivos → Puertos (COM y LPT)
```

### Compatibilidad de Firmware

Rayforge está diseñado para firmware compatible con GRBL. Asegúrate de que tu controlador ejecute:

- **GRBL 1.1** (más común, recomendado)
- **GRBL 0.9** (más antiguo, puede tener funciones limitadas)
- **grblHAL** (fork moderno de GRBL, soportado)

Otros tipos de firmware (Marlin, Smoothieware) no están actualmente soportados vía el controlador GRBL.

### Chipsets USB a Serie

Chipsets comunes y sus controladores:

| Chipset          | Linux | Windows | macOS |
| ---------------- | ----- | ------- | ----- |
| **CH340/CH341**  | Controlador de kernel integrado | [Controlador CH341SER](http://www.wch.cn/downloads/) | Requiere controlador |
| **FTDI FT232**   | Controlador de kernel integrado | Integrado (Windows 10+) | Integrado |
| **CP2102 (SiLabs)** | Controlador de kernel integrado | Integrado (Windows 10+) | Integrado |

---

## ¿Todavía Tienes Problemas?

Si has probado todo lo anterior y todavía no puedes conectar:

1. **Revisa los issues de GitHub** - Alguien puede haber reportado el mismo problema
2. **Crea un reporte de issue detallado** con:
   - Sistema operativo y versión
   - Versión de Rayforge (Snap/Flatpak/AppImage/código fuente)
   - Modelo de placa controladora y versión de firmware
   - Chipset USB (revisa Administrador de Dispositivos en Windows o `lsusb` en Linux)
   - Mensajes de error completos y registros de depuración
3. **Prueba con otra aplicación** - Prueba conectar con un terminal serie (minicom, PuTTY, Monitor Serie de Arduino) para verificar que el hardware funciona

---

## Páginas Relacionadas

- [Permisos de Snap](snap-permissions) - Configuración de permisos de Snap en Linux
- [Modo Depuración](debug) - Habilitar registro de diagnóstico
- [Ajustes Generales](../machine/general) - Guía de configuración de máquina
- [Ajustes de Dispositivo](../machine/device) - Referencia de configuración de GRBL
