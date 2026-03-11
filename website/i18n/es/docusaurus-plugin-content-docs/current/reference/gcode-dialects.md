# Soporte de Dialectos de Código G

Rayforge soporta múltiples dialectos de código G para trabajar con diferentes firmware de controlador.

## Dialectos Soportados

Rayforge actualmente soporta estos dialectos de código G:

| Dialecto                                      | Firmware     | Uso Común                               |
| --------------------------------------------- | ------------ | --------------------------------------- |
| **GRBL (universal)**                          | GRBL 1.1+    | Láseres de diodo, CNC de aficionado     |
| **GRBL (sin eje Z)**                          | GRBL 1.1+    | Cortadores láser 2D sin Z               |
| **GRBL Dinámico (Consciente de Profundidad)** | GRBL 1.1+    | Grabado láser consciente de profundidad |
| **GRBL Dinámico (sin eje Z)**                 | GRBL 1.1+    | Grabado láser consciente de profundidad |
| **Mach4 (M67 Analog)**                        | Mach4        | Grabado ráster de alta velocidad        |
| **Smoothieware**                              | Smoothieware | Cortadores láser, CNC                   |
| **Marlin**                                    | Marlin 2.0+  | Impresoras 3D con láser                 |

:::note Dialectos Recomendados
:::

**GRBL (universal)** es el dialecto más probado y recomendado para aplicaciones láser estándar.

**GRBL Dinámico (Consciente de Profundidad)** es recomendado para grabado láser consciente de profundidad donde la potencia varía durante los cortes (ej., grabado de profundidad variable).

---

## Mach4 (M67 Analog)

El dialecto **Mach4 (M67 Analog)** está diseñado para grabado ráster de alta velocidad con controladores Mach4. Utiliza el comando M67 con salida analógica para un control preciso de la potencia del láser.

### Características Principales

- **Salida Analógica M67**: Utiliza `M67 E0 Q<0-255>` para la potencia del láser en lugar de comandos S en línea
- **Presión de Búfer Reducida**: Al separar los comandos de potencia de los comandos de movimiento, el búfer del controlador sufre menos estrés durante operaciones de alta velocidad
- **Ráster de Alta Velocidad**: Optimizado para operaciones de grabado ráster rápidas

### Cuándo Usar

Use este dialecto cuando:

- Tenga un controlador Mach4 con capacidad de salida analógica
- Necesite grabado ráster de alta velocidad
- Su controlador experimente desbordamiento de búfer con comandos S en línea estándar

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
2. Haz clic en el ícono **Copiar** en un dialecto integrado para crear un nuevo dialecto personalizado
3. Edita los ajustes del dialecto según sea necesario
4. Guarda tu dialecto personalizado

Los dialectos personalizados se almacenan en tu directorio de configuración y pueden compartirse.

---

## Páginas Relacionadas

- [Exportando Código G](../files/exporting) - Ajustes de exportación
- [Compatibilidad de Firmware](firmware) - Versiones de firmware
- [Ajustes de Dispositivo](../machine/device) - Configuración de GRBL
- [Macros y Hooks](../machine/hooks-macros) - Inyección de código G personalizado
