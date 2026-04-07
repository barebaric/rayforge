# Soporte de Dialectos de Código G

Rayforge soporta múltiples dialectos de código G para trabajar con diferentes
firmware de controlador.

## Dialectos Soportados

Rayforge actualmente soporta estos dialectos de código G:

| Dialecto                                      | Firmware     | Uso Común                                    |
| --------------------------------------------- | ------------ | -------------------------------------------- |
| **Grbl (Compat)**                             | GRBL 1.1+    | Láseres de diodo, CNC de aficionado          |
| **Grbl (Compat, sin eje Z)**                  | GRBL 1.1+    | Cortadores láser 2D sin Z                    |
| **Grbl Raster**                               | GRBL 1.1+    | Optimizado para trabajo raster               |
| **GRBL Dinámico (Consciente de Profundidad)** | GRBL 1.1+    | Grabado láser consciente de profundidad      |
| **GRBL Dinámico (sin eje Z)**                 | GRBL 1.1+    | Grabado láser consciente de profundidad      |
| **Mach4 (M67 Analog)**                        | Mach4        | Grabado ráster de alta velocidad             |
| **Smoothieware**                              | Smoothieware | Cortadores láser, CNC                        |
| **Marlin**                                    | Marlin 2.0+  | Impresoras 3D con láser                      |

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

## Páginas Relacionadas

- [Exportando Código G](../files/exporting) - Ajustes de exportación
- [Compatibilidad de Firmware](firmware) - Versiones de firmware
- [Ajustes de Dispositivo](../machine/device) - Configuración de GRBL
- [Macros y Hooks](../machine/hooks-macros) - Inyección de código G personalizado
