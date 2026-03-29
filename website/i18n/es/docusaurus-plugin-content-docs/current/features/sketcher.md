# Diseñador paramétrico 2D

El Diseñador paramétrico 2D es una función potente de Rayforge que le permite
crear y editar diseños 2D precisos basados en restricciones directamente dentro
de la aplicación. Esta función le permite diseñar piezas personalizadas desde
cero sin necesidad de software CAD externo.

## Descripción general

El diseñador proporciona un conjunto completo de herramientas para crear formas
geométricas y aplicar restricciones paramétricas para definir relaciones precisas
entre los elementos. Este enfoque garantiza que sus diseños mantengan la
geometría prevista incluso cuando se modifican las dimensiones.

## Creación y edición de bocetos

### Crear un nuevo boceto

1. Haga clic en el botón "Nuevo boceto" en la barra de herramientas o use el
   menú principal
2. Se abrirá un nuevo espacio de trabajo vacío con la interfaz del editor de
   bocetos
3. Comience a crear geometría con las herramientas de dibujo del menú circular
   o los atajos de teclado
4. Aplique restricciones para definir las relaciones entre los elementos
5. Haga clic en "Finalizar boceto" para guardar su trabajo y volver al espacio
   de trabajo principal

### Editar bocetos existentes

1. Haga doble clic en una pieza de trabajo basada en boceto en el espacio de
   trabajo principal
2. Alternativamente, seleccione un boceto y elija "Editar boceto" en el menú
   contextual
3. Realice sus modificaciones con las mismas herramientas y restricciones
4. Haga clic en "Finalizar boceto" para guardar los cambios o en "Cancelar
   boceto" para descartarlos

## Creación de geometría 2D

El diseñador permite crear los siguientes elementos geométricos básicos:

- **Trazados (líneas y curvas Bézier)**: Dibuje líneas rectas y curvas Bézier
  suaves con la herramienta de trazado unificada. Haga clic para colocar puntos,
  arrastre para crear tiradores Bézier.
- **Arcos**: Dibuje arcos especificando un punto central, un punto de inicio y
  un punto final
- **Elipses**: Cree elipses (y círculos) definiendo un punto central y
  arrastrando para ajustar el tamaño y la proporción. Mantenga pulsado `Ctrl`
  mientras arrastra para restringir a un círculo perfecto.
- **Rectángulos**: Dibuje rectángulos especificando dos esquinas opuestas
- **Rectángulos redondeados**: Dibuje rectángulos con esquinas redondeadas
- **Cuadros de texto**: Añada elementos de texto a su boceto
- **Rellenos**: Rellene regiones cerradas para crear áreas sólidas

Estos elementos forman la base de sus diseños 2D y pueden combinarse para crear
formas complejas. Los rellenos son especialmente útiles para crear regiones
sólidas que se grabarán o cortarán como una sola pieza.

## Trabajar con curvas Bézier

La herramienta de trazado admite curvas Bézier para crear formas suaves y
orgánicas:

### Dibujar curvas Bézier

1. Seleccione la herramienta de trazado en el menú circular o use el atajo de
   teclado
2. Haga clic para colocar puntos; cada clic crea un nuevo punto
3. Arrastre tras hacer clic para crear tiradores Bézier y obtener curvas suaves
4. Siga añadiendo puntos para construir su trazado
5. Pulse Escape o haga doble clic para finalizar el trazado

### Editar curvas Bézier

- **Mover puntos**: Haga clic y arrastre cualquier punto para reposicionarlo
- **Ajustar tiradores**: Arrastre los extremos de los tiradores para modificar
  la forma de la curva
- **Conectar a puntos existentes**: Al editar un trazado, puede ajustarse a los
  puntos existentes de su boceto
- **Suavizar/simetrizar**: Los puntos conectados por una restricción de
  coincidencia pueden suavizarse (tangente continua) o simetrizarse (tiradores
  reflejados)

### Convertir curvas en líneas

Use la **herramienta de enderezamiento** para convertir curvas Bézier en líneas
rectas. Esto es útil cuando necesita geometría limpia y sencilla. Seleccione los
segmentos Bézier que desea convertir y aplique la acción de enderezamiento.

## Sistema de restricciones paramétricas

El sistema de restricciones es el núcleo del diseñador paramétrico, permitiéndole
definir relaciones geométricas precisas:

### Restricciones geométricas

- **Coincidencia**: Fuerza dos puntos a ocupar la misma posición
- **Vertical**: Restringe una línea para que sea perfectamente vertical
- **Horizontal**: Restringe una línea para que sea perfectamente horizontal
- **Tangente**: Hace que una línea sea tangente a un círculo o arco
- **Perpendicular**: Fuerza dos líneas, una línea y un arco/círculo, o dos
  arcos/círculos a encontrarse en un ángulo de 90 grados
- **Punto sobre línea/forma**: Restringe un punto para que se encuentre sobre
  una línea, arco o círculo
- **Colineal**: Fuerza dos o más líneas a encontrarse sobre la misma línea
  infinita
- **Simetría**: Crea relaciones simétricas entre elementos. Admite dos modos:
  - **Simetría de punto**: Seleccione 3 puntos (el primero es el centro)
  - **Simetría de línea**: Seleccione 2 puntos y 1 línea (la línea es el eje)

### Restricciones dimensionales

- **Distancia**: Establece la distancia exacta entre dos puntos o a lo largo de
  una línea
- **Diámetro**: Define el diámetro de un círculo
- **Radio**: Establece el radio de un círculo o arco
- **Ángulo**: Exige un ángulo específico entre dos líneas
- **Relación de aspecto**: Fuerza la proporción entre dos distancias a ser igual
  a un valor especificado
- **Igual longitud/radio**: Fuerza múltiples elementos (líneas, arcos, elipses
  o círculos) a tener la misma longitud o radio
- **Igual distancia**: Hace que dos segmentos de línea tengan la misma longitud
  (diferente de Igual longitud/radio, que también puede aplicarse a arcos y
  círculos)

## Interfaz del menú circular

El diseñador incluye un menú circular contextual que proporciona acceso rápido a
todas las herramientas de dibujo y restricción. Este menú radial aparece al
hacer clic derecho en el espacio de trabajo del boceto y se adapta según su
contexto y selección actuales.

Los elementos del menú circular muestran dinámicamente las opciones disponibles
según lo que tenga seleccionado. Por ejemplo, al hacer clic en un espacio vacío,
verá herramientas de dibujo. Al hacer clic sobre geometría seleccionada, verá las
restricciones aplicables.

![Menú circular del diseñador](/screenshots/sketcher-pie-menu.png)

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
- `G+G`: Herramienta de cuadrícula (alternar visibilidad de la cuadrícula)
- `G+N`: Alternar modo construcción en la selección

### Atajos de acciones
- `C+H`: Añadir chaflán en la esquina
- `C+F`: Añadir redondeo en la esquina
- `C+S`: Enderezar las curvas Bézier seleccionadas a líneas

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
- `Delete`: Eliminar los elementos seleccionados
- `Escape`: Cancelar la operación actual o deseleccionar
- `F`: Ajustar la vista al contenido

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
el diseñador ofrecerá bloquearla en su posición. Esto ayuda a mantener el boceto
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

## Importación y exportación

### Exportar objetos

Puede exportar cualquier pieza de trabajo seleccionada a varios formatos
vectoriales:

1. Seleccione una pieza de trabajo en el lienzo
2. Elija **Objeto → Exportar objeto...** (o haga clic derecho y seleccione desde
   el menú contextual)
3. Elija el formato de exportación:
   - **RFS (.rfs)**: Formato nativo de boceto paramétrico de Rayforge; conserva
     todas las restricciones y puede reimportarse para editar
   - **SVG (.svg)**: Formato vectorial estándar; ampliamente compatible con
     software de diseño
   - **DXF (.dxf)**: Formato de intercambio CAD; compatible con la mayoría de
     aplicaciones CAD

### Guardar bocetos

Puede guardar sus bocetos 2D en archivos para reutilizarlos en otros proyectos.
Todas las restricciones paramétricas se conservan al guardar, garantizando que
sus diseños mantengan sus relaciones geométricas.

### Importar bocetos

Los bocetos guardados pueden importarse en cualquier espacio de trabajo,
permitiéndole crear una biblioteca de elementos de diseño de uso común. El
proceso de importación mantiene todas las restricciones y relaciones
dimensionales.

## Consejos de flujo de trabajo

1. **Comience con geometría aproximada**: Cree primero formas básicas y luego
   refínelas con restricciones
2. **Use restricciones desde el principio**: Aplique restricciones mientras
   construye para mantener la intención del diseño
3. **Verifique el estado de las restricciones**: El sistema indica cuándo los
   bocetos están completamente restringidos
4. **Vigile los conflictos**: Las restricciones en conflicto se resaltan en rojo
   y se muestran en el panel de restricciones para facilitar su identificación
5. **Aproveche la simetría**: Las restricciones de simetría pueden acelerar
   significativamente los diseños complejos
6. **Use la cuadrícula**: Active la cuadrícula para una alineación precisa y use
   Ctrl para ajustar a la cuadrícula
7. **Itere y refínelo**: No dude en modificar las restricciones para obtener el
   resultado deseado

## Funciones de edición

- **Soporte completo de deshacer/rehacer**: El estado completo del boceto se
  guarda con cada operación
- **Cursor dinámico**: El cursor cambia para reflejar la herramienta de dibujo
  activa
- **Visualización de restricciones**: Las restricciones aplicadas se indican
  claramente en la interfaz
- **Actualizaciones en tiempo real**: Los cambios en las restricciones actualizan
  inmediatamente la geometría
- **Edición con doble clic**: Hacer doble clic en restricciones dimensionales
  (Distancia, Radio, Diámetro, Ángulo, Relación de aspecto) abre un diálogo
  para editar sus valores
- **Expresiones paramétricas**: Las restricciones dimensionales admiten
  expresiones, lo que permite calcular valores a partir de otros parámetros
  (p. ej., `width/2` para un radio que sea la mitad de la anchura)
