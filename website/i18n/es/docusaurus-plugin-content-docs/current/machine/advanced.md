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

## Invertir Ejes

Estos ajustes invierten la dirección de los movimientos de los ejes.

### Invertir Eje X

Invierte la dirección del eje X. Cuando está habilitado, X positivo se mueve a la izquierda en lugar de a la derecha.

### Invertir Eje Y

Invierte la dirección del eje Y. Cuando está habilitado, Y positivo se mueve hacia abajo en lugar de hacia arriba.

:::info
Invertir ejes es útil cuando:
- El sistema de coordenadas de tu máquina no coincide con el comportamiento esperado
- Has cableado tus motores al revés
- Quieres coincidir con el comportamiento de otra máquina
:::

## Ver También

- [Ajustes de Hardware](hardware) - Configuración de origen de ejes
- [Ajustes de Dispositivo](device) - Ajustes de dirección de ejes de GRBL
