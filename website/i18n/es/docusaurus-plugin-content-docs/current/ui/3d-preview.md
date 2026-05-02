# Vista 3D

La vista 3D te permite visualizar tus trayectorias de código G y simular la ejecución
del trabajo antes de enviarlas a tu máquina.

![Vista Previa 3D](/screenshots/main-3d.png)

## Abriendo la Vista 3D

Accede a la vista 3D:

- **Menú**: Ver → Vista 3D
- **Teclado**: <kbd>F12</kbd>

## Navegación

### Controles del Ratón

- **Rotar**: Clic izquierdo y arrastrar (eje Z), clic central y arrastrar (órbita de 3 ejes)
- **Desplazar**: <kbd>shift</kbd> + clic central y arrastrar
- **Zoom**: Rueda de desplazamiento

### Preajustes de Vista

Ángulos de cámara rápidos:

- **Superior** (<kbd>1</kbd>): Vista desde arriba
- **Frontal** (<kbd>2</kbd>): Elevación frontal
- **Derecha** (<kbd>3</kbd>): Vista lateral derecha
- **Izquierda** (<kbd>4</kbd>): Vista lateral izquierda
- **Posterior** (<kbd>5</kbd>): Elevación posterior
- **Isométrica** (<kbd>7</kbd>): Vista isométrica 3D

## Pantalla del Sistema de Coordenadas de Trabajo

La vista 3D visualiza el Sistema de Coordenadas de Trabajo (WCS) activo
de manera diferente al lienzo 2D:

### Cuadrícula y Ejes

- **Pantalla aislada**: La cuadrícula y los ejes aparecen como si el origen WCS fuera
  el origen del mundo
- **Desplazamiento aplicado**: Toda la cuadrícula se desplaza para alinearse con el desplazamiento
  WCS seleccionado
- **Etiquetas relativas al WCS**: Las etiquetas de coordenadas muestran posiciones relativas al
  origen WCS, no al origen de máquina

Esta pantalla "en aislamiento" facilita entender dónde se ejecutará tu trabajo
relativo al sistema de coordenadas de trabajo seleccionado, sin confundirse
por la posición absoluta de la máquina.

### Cambiando WCS

La vista 3D se actualiza automáticamente cuando cambias el WCS activo:
- Selecciona un WCS diferente del menú desplegable de la barra de herramientas
- La cuadrícula y los ejes se desplazan para reflejar el nuevo origen WCS
- Las etiquetas se actualizan para mostrar coordenadas relativas al nuevo WCS

:::tip WCS en Vista 3D
La vista 3D muestra tus trayectorias relativas al WCS seleccionado. Cuando
cambias WCS, verás las trayectorias parecer moverse porque el punto de referencia
(la cuadrícula) ha cambiado, no porque las trayectorias mismas se movieron.
:::


## Opciones de Pantalla

Los controles de visibilidad están ubicados como botones superpuestos en la esquina
superior derecha del lienzo 3D:

- **Modelo**: Alternar la visibilidad del modelo 3D de la máquina
- **Movimientos de desplazamiento**: Alternar la visibilidad de los movimientos rápidos
- **Zonas prohibidas**: Alternar la visibilidad de las zonas prohibidas

### Visualización de Trayectoria

Personaliza lo que ves:

- **Mostrar Movimientos Rápidos**: Mostrar movimientos de desplazamiento (líneas punteadas)
- **Mostrar Movimientos de Trabajo**: Mostrar movimientos de corte/grabado (líneas sólidas)
- **Colorear por Operación**: Diferentes colores para cada operación

:::tip Colores por Láser
Al usar máquinas con múltiples cabezales láser, cada láser puede tener sus propios
colores de corte y raster configurados en [Ajustes de Láser](../machine/laser).
Esto facilita identificar qué láser realizará cada operación.
:::

### Modelo de Cabezal Láser

La vista 3D renderiza un modelo de tu cabezal láser que se mueve a lo largo de
la trayectoria durante la simulación. Puedes asignar un modelo 3D a cada cabezal
láser en la página de [Ajustes de Láser](../machine/laser) en Configuración de
Máquina. La escala, rotación y distancia focal del modelo se pueden ajustar para
coincidir con tu configuración física.

Durante la simulación, se dibuja un rayo láser brillante desde el cabezal hacia
abajo cuando el láser está activo.

## Simulación

La vista 3D incluye un simulador integrado con controles de reproducción
superpuestos en la parte inferior del lienzo.

### Controles de Reproducción

- **Reproducir/Pausar** (<kbd>espacio</kbd>): Animar la ejecución de la trayectoria
- **Paso Adelante/Atrás**: Avanzar o retroceder una operación a la vez
- **Velocidad**: Ciclar entre velocidades de reproducción (1x, 2x, 4x, 8x, 16x)
- **Deslizador de línea de tiempo**: Arrastra para recorrer el trabajo

### Visor de Código G Sincronizado

La simulación se mantiene sincronizada con el visor de código G en el panel inferior.
Recorrer la simulación resalta la línea correspondiente en el visor de código G,
y hacer clic en una línea en el visor de código G salta la simulación a ese punto.

### Visibilidad de Capas

Alterna la visibilidad de capas individuales:

- Haz clic en el nombre de una capa para mostrarla u ocultarla
- Enfócate en capas específicas para inspección

## Lista de Verificación

Antes de enviar a la máquina, verifica:

- [ ] La trayectoria está completa sin segmentos faltantes
- [ ] Las operaciones de grabado se ejecutan antes que los cortes
- [ ] El trabajo comienza en la posición esperada
- [ ] Las pestañas de sujeción están en las ubicaciones correctas

Algunas comprobaciones adicionales se realizan automáticamente. Cuando ejecutas o
exportas un trabajo, Rayforge ejecuta [comprobaciones de sanity](../features/sanity-checks)
que verifican los límites de la máquina, las fronteras del área de trabajo y las
colisiones con zonas prohibidas.

## Consejos de Rendimiento

Para trabajos grandes o complejos:

1. Oculta movimientos rápidos para enfocarte solo en los movimientos de trabajo
2. Reduce el número de capas visibles
3. Cierra otras aplicaciones para liberar recursos de GPU

## Solución de Problemas

### La vista previa está en blanco o negra

- Verifica que las operaciones estén habilitadas
- Verifica que los objetos tengan operaciones asignadas

### Vista previa lenta o entrecortada

- Oculta movimientos rápidos
- Oculta modelos 3D
- Reduce el número de capas visibles

---

**Páginas Relacionadas:**

- [Sistemas de Coordenadas de Trabajo](../general-info/coordinate-systems) - WCS
- [Ventana Principal](main-window) - Resumen de la interfaz principal
- [Ajustes](settings) - Preferencias de la aplicación
