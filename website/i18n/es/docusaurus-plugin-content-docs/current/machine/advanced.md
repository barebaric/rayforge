# Ajustes Avanzados

La página Avanzado en Configuración de Máquina contiene opciones de configuración adicionales para
casos de uso especializados.

![Ajustes Avanzados](/screenshots/machine-settings-advanced.webp)

## Comportamiento de Conexión

Ajustes que controlan cómo Rayforge interactúa con tu máquina durante la conexión.

### Home al Conectar

Cuando está habilitado, Rayforge envía automáticamente un comando de homing ($H) al conectar a la
máquina.

- **Habilitar si**: Tu máquina tiene interruptores de límite confiables
- **Deshabilitar si**: Tu máquina no tiene interruptores de límite o el homing no es confiable

### Limpiar Alarmas al Conectar

Cuando está habilitado, Rayforge limpia automáticamente cualquier estado de alarma al conectar.

- **Habilitar si**: Tu máquina frecuentemente inicia en estado de alarma
- **Deshabilitar si**: Quieres investigar manualmente las alarmas antes de limpiarlas

### Permitir Homing de Eje Individual

Cuando está habilitado, puedes hacer homing de ejes individuales independientemente (X, Y o Z) en
lugar de requerir que todos los ejes hagan homing juntos. Esto es útil para máquinas donde un eje ya
puede estar posicionado correctamente.

### Referenciado de Z

Elige cómo encuentra el eje Z su posición de referencia al hacer homing. Este ajuste solo aparece
cuando el controlador de la máquina admite más de un método.

- **Interruptor de origen**: Z se mueve a su interruptor de origen, igual que los ejes X e Y
- **Sonda**: Z usa la sonda de autoenfoque del controlador para encontrar la superficie de trabajo y
  luego se mueve a la distancia de enfoque configurada

## Ajustes de Arcos y Curvas

Ajustes para controlar cómo las rutas curvas se convierten en movimientos de G-code.

### Soportar Arcos

Cuando está habilitado, Rayforge genera comandos de arco (G2/G3) para rutas curvas en lugar de
dividirlas en muchos movimientos lineales pequeños. Esto produce G-code más compacto y movimiento
más suave en la mayoría de los controladores.

Cuando está deshabilitado, todas las curvas se convierten en segmentos lineales (comandos G1), lo
que proporciona máxima compatibilidad con controladores que no soportan arcos.

### Soportar Curvas Bézier

Cuando está habilitado, Rayforge genera comandos cúbicos Bézier nativos (como el comando G5 usado
por LinuxCNC) para rutas curvas. Esto produce movimiento muy suave y G-code compacto en
controladores que lo soportan. Debes deshabilitar esta opción si el firmware de tu máquina no
entiende comandos Bézier, en cuyo caso las curvas se descompondrán en segmentos lineales.

### Tolerancia de Arco y Curva

Este ajuste controla la desviación máxima permitida al ajustar arcos y curvas a rutas curvas,
especificada en milímetros. Un valor más pequeño produce rutas más precisas pero puede requerir más
comandos. Un valor más grande permite más desviación pero genera menos comandos.

Valores típicos van de 0.01mm para trabajo de precisión a 0.1mm para procesamiento más rápido.

## Desplazamiento de Escaneo Bidireccional por Velocidad {#bidirectional-scan-offset-by-speed}

La tabla de desplazamiento de escaneo bidireccional por velocidad corrige la desalineación
horizontal entre las pasadas raster de izquierda a derecha y de derecha a izquierda. En muchas
máquinas este sesgo no es constante: cuanto más rápido se mueve el cabezal láser, más se desvían las
pasadas de derecha a izquierda de su alineación, lo que se ve como bandas que empeoran a velocidades
de grabado más altas.

![Desplazamiento de Escaneo Bidireccional por Velocidad](/screenshots/machine-settings-advanced-bidir-offset-table.webp)

Cada fila de la tabla asocia una velocidad de grabado con el desplazamiento, en milímetros, que las
pasadas de derecha a izquierda necesitan a esa velocidad:

- Usa **Añadir Fila** para añadir una fila y el botón de papelera de una fila para eliminarla
- Los valores positivos desplazan las pasadas de derecha a izquierda hacia la derecha, los negativos
  hacia la izquierda
- Las filas se mantienen ordenadas por velocidad; si dos filas tienen la misma velocidad, la última
  es la que vale

Cuando se ejecuta un paso de grabado, Rayforge busca la velocidad del paso en esta tabla e interpola
el desplazamiento linealmente entre las filas circundantes. Las velocidades por debajo de la primera
fila o por encima de la última usan el desplazamiento de la fila más cercana, y una tabla vacía
significa que todas las velocidades usan un desplazamiento de 0.

**Sin corrección** — la desalineación crece con la velocidad de grabado:

![Pasadas raster desalineadas a 1000, 4000 y 8000 mm/min](/images/bidir-scan-offset-uncorrected.webp)

**Con la tabla de desplazamiento aplicada:**

![Pasadas raster alineadas a todas las velocidades](/images/bidir-scan-offset-table-applied.webp)

La tabla solo se aplica a los pasos de grabado cuyo propio **Desplazamiento de Escaneo
Bidireccional** es 0; un desplazamiento por paso distinto de cero siempre tiene prioridad. Consulta
[Desplazamiento de Escaneo Bidireccional](../features/operations/engrave.md#bidirectional-scan-offset)
para ver cómo es la desalineación y cómo calibrarla.

## Ver También

- [Ajustes de Hardware](hardware) - Configuración de origen de ejes e inversión
- [Ajustes de Dispositivo](device) - Ajustes específicos de GRBL
