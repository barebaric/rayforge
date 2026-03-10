# Ajustes Avanzados

La página Avanzado en Configuración de Máquina contiene opciones de configuración adicionales para casos de uso especializados.

![Ajustes Avanzados](/screenshots/machine-advanced.png)

## Comportamiento de Conexión

Ajustes que controlan cómo Rayforge interactúa con tu máquina durante la conexión.

### Home al Conectar

Cuando está habilitado, Rayforge envía automáticamente un comando de homing ($H) al conectar a la máquina.

- **Habilitar si**: Tu máquina tiene interruptores de límite confiables
- **Deshabilitar si**: Tu máquina no tiene interruptores de límite o el homing no es confiable

### Limpiar Alarmas al Conectar

Cuando está habilitado, Rayforge limpia automáticamente cualquier estado de alarma al conectar.

- **Habilitar si**: Tu máquina frecuentemente inicia en estado de alarma
- **Deshabilitar si**: Quieres investigar manualmente las alarmas antes de limpiarlas

### Permitir Homing de Eje Individual

Cuando está habilitado, puedes hacer homing de ejes individuales independientemente (X, Y o Z) en lugar de requerir que todos los ejes hagan homing juntos. Esto es útil para máquinas donde un eje ya puede estar posicionado correctamente.

## Ajustes de Arcos

Ajustes para controlar cómo las rutas curvas se convierten en movimientos de G-code.

### Soportar Arcos

Cuando está habilitado, Rayforge genera comandos de arco (G2/G3) para rutas curvas en lugar de dividirlas en muchos movimientos lineales pequeños. Esto produce G-code más compacto y movimiento más suave en la mayoría de los controladores.

Cuando está deshabilitado, todas las curvas se convierten en segmentos lineales (comandos G1), lo que proporciona máxima compatibilidad con controladores que no soportan arcos.

### Tolerancia de Arco

Este ajuste controla la desviación máxima permitida al ajustar arcos a rutas curvas, especificada en milímetros. Un valor más pequeño produce arcos más precisos pero puede requerir más comandos de arco. Un valor más grande permite más desviación pero genera menos comandos.

Valores típicos van de 0.01mm para trabajo de precisión a 0.1mm para procesamiento más rápido.

## Ver También

- [Ajustes de Hardware](hardware) - Configuración de origen de ejes e inversión
- [Ajustes de Dispositivo](device) - Ajustes específicos de GRBL
