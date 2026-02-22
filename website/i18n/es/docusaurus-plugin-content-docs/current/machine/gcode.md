# Ajustes de Código G

La página de Código G en Configuración de Máquina configura cómo Rayforge genera código G para tu máquina.

![Ajustes de Código G](/screenshots/machine-gcode.png)

## Dialecto de Código G

Selecciona el dialecto de código G que coincide con el firmware de tu controlador. Diferentes controladores usan comandos y formatos ligeramente diferentes.

### Dialectos Disponibles

- **GRBL**: Más común para cortadores láser de aficionado. Usa M3/M5 para control del láser.
- **Smoothieware**: Para Smoothieboard y controladores similares.
- **Marlin**: Para controladores basados en Marlin.
- **GRBL-compatible**: Para controladores que mayormente siguen la sintaxis GRBL.

:::info
El dialecto afecta cómo la potencia del láser, los movimientos y otros comandos se formatean en el código G de salida.
:::

## Código G Personalizado

Puedes personalizar el código G que Rayforge genera en puntos específicos del trabajo.

### Inicio del Programa

Comandos de código G ejecutados al comienzo de cada trabajo, antes de cualquier operación de corte.

Usos comunes:
- Establecer unidades (G21 para mm)
- Establecer modo de posicionamiento (G90 para absoluto)
- Inicializar el estado de la máquina

### Fin del Programa

Comandos de código G ejecutados al final de cada trabajo, después de todas las operaciones de corte.

Usos comunes:
- Apagar láser (M5)
- Volver al origen (G0 X0 Y0)
- Estacionar la cabeza

### Cambio de Herramienta

Comandos de código G ejecutados al cambiar entre cabezales láser (para máquinas con múltiples láseres).

## Ver También

- [Fundamentos de Código G](../general-info/gcode-basics) - Entendiendo el código G
- [Dialectos de Código G](../reference/gcode-dialects) - Diferencias detalladas de dialectos
- [Hooks y Macros](hooks-macros) - Puntos de inyección de código G personalizado
