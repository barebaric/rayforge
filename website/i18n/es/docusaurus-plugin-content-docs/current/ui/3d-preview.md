# Vista Previa 3D

La ventana de vista previa 3D te permite visualizar tus trayectorias de código G antes
de enviarlas a tu máquina. Esta potente función te ayuda a detectar errores
y verificar la configuración de tu trabajo.

![Vista Previa 3D](/screenshots/main-3d.png)

## Abriendo la Vista Previa 3D

Accede a la vista previa 3D:

- **Menú**: Ver → Vista Previa 3D
- **Teclado**: <kbd>ctrl+3</kbd>
- **Después de generar código G**: Se abre automáticamente (configurable)

## Navegación

### Controles del Ratón

- **Rotar**: Clic izquierdo y arrastrar
- **Desplazar**: Clic derecho y arrastrar, o clic central y arrastrar
- **Zoom**: Rueda de desplazamiento, o <kbd>ctrl</kbd> + clic izquierdo y arrastrar

### Controles de Teclado

- <kbd>r</kbd>: Restablecer cámara a vista predeterminada
- <kbd>inicio</kbd>: Restablecer zoom y posición
- <kbd>f</kbd>: Ajustar vista a la trayectoria
- Teclas de flecha: Rotar cámara

### Preajustes de Vista

Ángulos de cámara rápidos:

- **Superior** (<kbd>1</kbd>): Vista desde arriba
- **Frontal** (<kbd>2</kbd>): Elevación frontal
- **Derecha** (<kbd>3</kbd>): Elevación lateral derecha
- **Isométrica** (<kbd>4</kbd>): Vista isométrica 3D

## Pantalla del Sistema de Coordenadas de Trabajo

La vista previa 3D visualiza el Sistema de Coordenadas de Trabajo (WCS) activo
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

La vista previa 3D se actualiza automáticamente cuando cambias el WCS activo:
- Selecciona un WCS diferente del menú desplegable de la barra de herramientas
- La cuadrícula y los ejes se desplazan para reflejar el nuevo origen WCS
- Las etiquetas se actualizan para mostrar coordenadas relativas al nuevo WCS

:::tip WCS en Vista Previa 3D
La vista previa 3D muestra tus trayectorias relativas al WCS seleccionado. Cuando
cambias WCS, verás las trayectorias parecer moverse porque el punto de referencia
(la cuadrícula) ha cambiado, no porque las trayectorias mismas se movieron.
:::


## Opciones de Pantalla

### Visualización de Trayectoria

Personaliza lo que ves:

- **Mostrar Movimientos Rápidos**: Mostrar movimientos de desplazamiento (líneas punteadas)
- **Mostrar Movimientos de Trabajo**: Mostrar movimientos de corte/grabado (líneas sólidas)
- **Colorear por Operación**: Diferentes colores para cada operación
- **Colorear por Potencia**: Degradado basado en la potencia del láser
- **Colorear por Velocidad**: Degradado basado en la velocidad de avance

### Visualización de Máquina

- **Mostrar Origen**: Mostrar punto de referencia (0,0)
- **Mostrar Área de Trabajo**: Mostrar límites de la máquina
- **Mostrar Cabezal Láser**: Mostrar indicador de posición actual

### Ajustes de Calidad

- **Ancho de Línea**: Grosor de las líneas de trayectoria
- **Anti-aliasing**: Renderizado de líneas suave (puede afectar el rendimiento)
- **Fondo**: Color claro, oscuro o personalizado

## Controles de Reproducción

Simula la ejecución del trabajo:

- **Reproducir/Pausar** (<kbd>espacio</kbd>): Animar la ejecución de la trayectoria
- **Velocidad**: Ajustar la velocidad de reproducción (0.5x - 10x)
- **Paso Adelante/Atrás**: Avanzar por comandos individuales de código G
- **Saltar a Posición**: Clic en la línea de tiempo para saltar a un punto específico

### Línea de Tiempo

La línea de tiempo muestra:

- Posición actual en el trabajo
- Límites de operación (segmentos coloreados)
- Tiempo estimado en cualquier punto

## Herramientas de Análisis

### Medición de Distancia

Mide distancias en 3D:

1. Habilita la herramienta de medición
2. Haz clic en dos puntos de la trayectoria
3. Ver la distancia en las unidades actuales

### Panel de Estadísticas

Ver estadísticas del trabajo:

- **Distancia Total**: Suma de todos los movimientos
- **Distancia de Trabajo**: Distancia de corte/grabado solo
- **Distancia Rápida**: Movimientos de desplazamiento solo
- **Tiempo Estimado**: Estimación de duración del trabajo
- **Cuadro Delimitador**: Dimensiones generales

### Visibilidad de Capas

Alterna la visibilidad de operaciones:

- Haz clic en el nombre de la operación para mostrar/ocultar
- Enfócate en operaciones específicas para inspección
- Aísla problemas sin regenerar código G

## Lista de Verificación de Verificación

Antes de enviar a la máquina, verifica:

- [ ] **La trayectoria está completa**: Sin segmentos faltantes
- [ ] **Dentro del área de trabajo**: Permanece dentro de los límites de la máquina
- [ ] **Orden de operación correcto**: Grabar antes de cortar
- [ ] **Sin colisiones**: La cabeza no golpea abrazaderas/fijaciones
- [ ] **Origen apropiado**: Comienza en la posición esperada
- [ ] **Posiciones de pestañas**: Pestañas de sujeción en las ubicaciones correctas (si se usan)

## Consejos de Rendimiento

Para trabajos grandes o complejos:

1. **Reduce detalle de líneas**: Baja la calidad de pantalla para renderizado más rápido
2. **Oculta movimientos rápidos**: Enfócate solo en movimientos de trabajo
3. **Deshabilita anti-aliasing**: Mejora la velocidad de fotogramas
4. **Cierra otras aplicaciones**: Libera recursos de GPU

## Solución de Problemas

### La vista previa está en blanco o negra

- Regenera código G (<kbd>ctrl+g</kbd>)
- Verifica que las operaciones estén habilitadas
- Verifica que los objetos tengan operaciones asignadas

### Vista previa lenta o entrecortada

- Reduce el ancho de línea
- Deshabilita anti-aliasing
- Oculta movimientos rápidos
- Actualiza controladores de gráficos

### Los colores no se muestran correctamente

- Revisa el ajuste de colorear por (operación/potencia/velocidad)
- Asegúrate de que las operaciones tengan diferentes colores asignados
- Restablece los ajustes de vista a los valores predeterminados

---

**Páginas Relacionadas:**

- [Sistemas de Coordenadas de Trabajo](../general-info/work-coordinate-systems) - WCS
- [Ventana Principal](main-window) - Resumen de la interfaz principal
- [Ajustes](settings) - Preferencias de la aplicación
