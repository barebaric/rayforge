---
description: "Herramientas del diseñador, atajos de teclado, menú circular, modo construcción, cuadrícula, ajuste, desplazamiento, chaflán y redondeo en Rayforge."
---

# Herramientas del diseñador

## Interfaz del menú circular

El diseñador incluye un menú circular contextual que proporciona acceso rápido a
todas las herramientas de dibujo y restricción. Este menú radial aparece al
hacer clic derecho en el espacio de trabajo del boceto y se adapta según su
contexto y selección actuales.

Los elementos del menú circular muestran dinámicamente las opciones disponibles
según lo que tenga seleccionado. Por ejemplo, al hacer clic en un espacio vacío,
verá herramientas de dibujo. Al hacer clic sobre geometría seleccionada, verá las
restricciones aplicables.

![Menú circular del diseñador](/screenshots/sketcher-pie-menu.webp)

## Atajos de teclado

El diseñador proporciona atajos de teclado para un flujo de trabajo eficiente:

### Atajos de herramientas

- `Space`: Herramienta de selección
- `G+P`: Herramienta de trazado (líneas y curvas Bézier)
- `G+A`: Herramienta de arco
- `G+C`: Herramienta de elipse
- `G+R`: Herramienta de rectángulo
- `G+O`: Herramienta de rectángulo redondeado
- `G+F`: Herramienta de relleno de área
- `G+T`: Herramienta de cuadro de texto
- `G+Y`: Herramienta de arreglo circular
- `G+W`: Herramienta de arreglo a lo largo de curva
- `G+G`: Herramienta de cuadrícula (alternar visibilidad de la cuadrícula)
- `G+N`: Alternar modo construcción en la selección

### Atajos de acciones

- `O+F`: Desplazar el contorno seleccionado
- `C+H`: Añadir chaflán en la esquina
- `C+F`: Añadir redondeo en la esquina
- `C+S`: Enderezar las curvas Bézier seleccionadas a líneas
- `M+V`: Reflejar la selección verticalmente
- `M+H`: Reflejar la selección horizontalmente
- `Ctrl+D`: Duplicar la selección en el lugar

### Atajos de restricciones

- `H`: Aplicar restricción Horizontal
- `V`: Aplicar restricción Vertical
- `N`: Aplicar restricción Perpendicular
- `T`: Aplicar restricción Tangente
- `E`: Aplicar restricción Igual
- `O` o `C`: Aplicar restricción de Alineación (Coincidencia)
- `S`: Aplicar restricción de Simetría
- `K+D`: Aplicar restricción de Distancia
- `K+R`: Aplicar restricción de Radio
- `K+O`: Aplicar restricción de Diámetro
- `K+A`: Aplicar restricción de Ángulo
- `K+X`: Aplicar restricción de Relación de aspecto

### Atajos generales

- `Ctrl+Z`: Deshacer
- `Ctrl+Y` o `Ctrl+Shift+Z`: Rehacer
- `Ctrl+D`: Duplicar los elementos seleccionados
- `Delete`: Eliminar los elementos seleccionados
- `Teclas de flecha`: Desplazar entidades seleccionadas (mantenga pulsado
  `Shift` para un paso mayor)
- `Escape`: Cancelar la operación actual o deseleccionar
- `F`: Ajustar la vista al contenido

## Espejo, Duplicar y Ajuste

Varias herramientas de transformación operan sobre la selección actual:

- **Reflejar Verticalmente / Horizontalmente** (`M+V` / `M+H`): refleja la
  selección en el lugar a través del centro de su cuadro delimitador. Las
  restricciones que cruzan el límite de la selección se eliminan; las
  restricciones internas se conservan.
- **Duplicar** (`Ctrl+D`): copia la selección en el lugar. Las copias obtienen
  IDs nuevos y restricciones internas reasignadas; solo las copias permanecen
  seleccionadas después. Deshacer las elimina.
- **Ajuste**: con las entidades seleccionadas, las **teclas de flecha** mueven
  la selección. Mantenga pulsado `Shift` para un paso de ajuste mayor.

Estas están disponibles en la barra de herramientas y el menú **Boceto**.

## Modo construcción

El modo construcción le permite marcar entidades como "geometría de construcción",
elementos auxiliares que guían su diseño pero que no forman parte del resultado
final. Las entidades de construcción se muestran de forma diferente
(normalmente como líneas discontinuas) y no se incluyen cuando el boceto se usa
para corte o grabado láser.

Para alternar el modo construcción:

- Seleccione una o más entidades
- Pulse `N` o `G+N`, o use la opción Construcción en el menú circular

Las entidades de construcción son útiles para:

- Crear líneas y círculos de referencia
- Definir geometría temporal para alineación
- Construir formas complejas a partir de un marco de guías

## Cuadrícula, ajuste y controles de visibilidad

### Herramienta de cuadrícula

La herramienta de cuadrícula proporciona una referencia visual para la alineación
y el tamaño:

- Active/desactive la cuadrícula con el botón de la herramienta o `G+G`
- La cuadrícula se adapta a su nivel de zoom para mantener un espaciado
  consistente

### Ajuste magnético

Mientras crea o mueve geometría, Rayforge atrae automáticamente el cursor hacia
los elementos cercanos: extremos, puntos medios de líneas, intersecciones y otros
puntos de referencia. Esto facilita la conexión precisa de formas sin colocar
manualmente cada punto. El indicador de ajuste se resalta cuando el cursor está
cerca de un objetivo de ajuste.

### Auto-restricción durante la creación

Muchas herramientas de dibujo aplican restricciones automáticamente al crear
geometría. Por ejemplo, al dibujar una línea cerca de la horizontal o vertical,
el diseñador ofrecerá bloquearla en su posición. La herramienta de
trazado también crea restricciones horizontales y verticales
automáticamente cuando las guías de ajuste muestran alineación durante el
dibujo. Esto ayuda a mantener el boceto
ordenado desde el principio, en lugar de corregirlo después.

### Controles de mostrar/ocultar

La barra de herramientas del diseñador incluye botones de alternancia para
controlar la visibilidad:

- **Mostrar/ocultar geometría de construcción**: Alterne la visibilidad de las
  entidades de construcción
- **Mostrar/ocultar restricciones**: Alterne la visibilidad de los marcadores de
  restricciones

Estos controles ayudan a reducir el desorden visual al trabajar con bocetos
complejos.

### Movimiento restringido al eje

Al arrastrar puntos o geometría, mantenga pulsado `Shift` para restringir el
movimiento al eje más cercano (horizontal o vertical). Esto es útil para
mantener la alineación mientras realiza ajustes.

## Desplazar contorno

La herramienta de desplazamiento agranda o encoge el contorno seleccionado una
distancia dada, o expande un trazado abierto en una ranura. Seleccione las
entidades que forman un contorno (o use doble clic para seleccionar la
geometría conectada), luego pulse `O+F`, o use la entrada **Desplazar** del
menú circular.

![Diálogo de desplazar contorno](/screenshots/addons-sketcher-offset-dialog.webp)

El diálogo pide la distancia de desplazamiento y muestra una vista previa en
vivo del resultado en el lienzo mientras escribe:

- Los **contornos cerrados** crecen con una distancia positiva y se encogen
  con una negativa. Se rechaza un desplazamiento que haría colapsar el
  contorno.
- Los **trazados abiertos** se convierten en un contorno de ranura cerrado del
  ancho indicado, con extremos redondeados.

![Contorno Bézier](/screenshots/addons-sketcher-offset-before.webp)
![Bézier desplazado a una ranura](/screenshots/addons-sketcher-offset-after.webp)

Al desplazar, el contorno seleccionado se reemplaza por el resultado:

- Los círculos, arcos y elipses aislados conservan su tipo de entidad y se
  actualizan en el lugar, por lo que siguen siendo editables y restringibles
  como antes.
- Las cadenas de segmentos conectados (incluidas las Bézier) se reemplazan por
  una entidad polígono. El polígono se edita como un todo: arrastre su punto
  central para moverlo y el punto de tirador para rotarlo o escalarlo de forma
  uniforme.

Si la selección contiene varios contornos desconectados, cada uno se desplaza
de forma independiente en un solo paso.

## Chaflán y redondeo

El diseñador proporciona herramientas para modificar las esquinas de su
geometría:

- **Chaflán**: Reemplaza una esquina aguda con un borde biselado. Seleccione un
  punto de unión (donde dos líneas se encuentran) y aplique la acción de
  chaflán.
- **Redondeo**: Reemplaza una esquina aguda con un borde redondeado. Seleccione
  un punto de unión (donde dos líneas se encuentran) y aplique la acción de
  redondeo.

Para usar chaflán o redondeo:

1. Seleccione un punto de unión donde dos líneas se encuentran
2. Pulse `C+H` para chaflán o `C+F` para redondeo
3. Use el menú circular o los atajos de teclado para aplicar la modificación
