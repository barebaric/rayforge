# Ajustes de Código G

La página de Código G en Configuración de Máquina configura cómo Rayforge genera código G para tu máquina.

![Ajustes de Código G](/screenshots/machine-gcode.png)

:::info
Esta página solo se muestra para máquinas que usan controladores basados en
G-code (p.ej. GRBL, Smoothieware, Marlin). Si tu máquina usa un controlador
no basado en G-code (p.ej. Ruida), esta página de ajustes está completamente
oculta.
:::

## Dialecto de Código G

Selecciona el dialecto de código G que coincide con el firmware de tu controlador. Diferentes controladores usan comandos y formatos ligeramente diferentes.

### Dialectos Disponibles

- **Grbl (Compat)**: Dialecto GRBL estándar para cortadoras láser de hobby. Usa M3/M5 para control del láser.
- **Grbl (Compat, no Z axis)**: Igual que Grbl (Compat) pero sin comandos de eje Z. Para máquinas solo 2D.
- **GRBL Dynamic**: Usa el modo de potencia láser dinámica de GRBL para grabado de potencia variable.
- **GRBL Dynamic (no Z axis)**: Modo dinámico sin comandos de eje Z.
- **LinuxCNC**: Para controladores LinuxCNC. Soporta curvas Bézier cúbicas (G5) nativas.
- **Smoothieware**: Para Smoothieboard y controladores similares.
- **Marlin**: Para controladores basados en Marlin.

:::info
El dialecto afecta cómo la potencia del láser, los movimientos y otros comandos se formatean en el código G de salida.
:::

## Preámbulo y Postscript del Dialecto

Cada dialecto incluye código G de preámbulo y postscript personalizable que se ejecuta al inicio y al final de los trabajos.

### Preámbulo

Comandos de código G ejecutados al comienzo de cada trabajo, antes de cualquier operación de corte. Usos comunes incluyen establecer unidades (G21 para mm), modo de posicionamiento (G90 para absoluto) e inicializar el estado de la máquina.

### Postscript

Comandos de código G ejecutados al final de cada trabajo, después de todas las operaciones de corte. Usos comunes incluyen apagar el láser (M5), volver al origen (G0 X0 Y0) y estacionar la cabeza.

## Ver También

- [Fundamentos de Código G](../general-info/gcode-basics) - Entendiendo el código G
- [Dialectos de Código G](../reference/gcode-dialects) - Diferencias detalladas de dialectos
- [Hooks y Macros](hooks-macros) - Puntos de inyección de código G personalizado
