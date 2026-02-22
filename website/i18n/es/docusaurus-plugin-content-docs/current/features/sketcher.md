# Dibujador 2D Paramétrico

El Dibujador 2D Paramétrico es una potente función en Rayforge que te permite crear y editar diseños 2D precisos basados en restricciones directamente dentro de la aplicación. Esta función te permite diseñar piezas personalizadas desde cero sin necesidad de software CAD externo.

## Resumen

El dibujador proporciona un conjunto completo de herramientas para crear formas geométricas y aplicar restricciones paramétricas para definir relaciones precisas entre elementos. Este enfoque asegura que tus diseños mantengan su geometría intencionada incluso cuando se modifican las dimensiones.

## Crear y Editar Bocetos

### Crear un Nuevo Boceto

1. Haz clic en el botón "Nuevo Boceto" en la barra de herramientas o usa el menú principal
2. Se abrirá un nuevo espacio de trabajo de boceto vacío con la interfaz del editor de bocetos
3. Comienza a crear geometría usando las herramientas de dibujo del menú circular o atajos de teclado
4. Aplica restricciones para definir relaciones entre elementos
5. Haz clic en "Finalizar Boceto" para guardar tu trabajo y volver al espacio de trabajo principal

### Editar Bocetos Existentes

1. Haz doble clic en una pieza de trabajo basada en boceto en el espacio de trabajo principal
2. Alternativamente, selecciona un boceto y elige "Editar Boceto" del menú contextual
3. Realiza tus modificaciones usando las mismas herramientas y restricciones
4. Haz clic en "Finalizar Boceto" para guardar cambios o "Cancelar Boceto" para descartarlos

## Crear Geometría 2D

El dibujador soporta crear los siguientes elementos geométricos básicos:

- **Líneas**: Dibuja segmentos de línea recta entre puntos
- **Círculos**: Crea círculos definiendo un punto central y radio
- **Arcos**: Dibuja arcos especificando un punto central, punto inicial y punto final
- **Rectángulos**: Dibuja rectángulos especificando dos esquinas opuestas
- **Rectángulos Redondeados**: Dibuja rectángulos con esquinas redondeadas
- **Cajas de Texto**: Añade elementos de texto a tu boceto
- **Rellenos**: Rellena regiones cerradas para crear áreas sólidas

Estos elementos forman la base de tus diseños 2D y pueden combinarse para crear formas complejas. Los rellenos son particularmente útiles para crear regiones sólidas que serán grabadas o cortadas como una sola pieza.

## Sistema de Restricciones Paramétricas

El sistema de restricciones es el núcleo del dibujador paramétrico, permitiéndote definir relaciones geométricas precisas:

### Restricciones Geométricas

- **Coincidente**: Fuerza dos puntos a ocupar la misma ubicación
- **Vertical**: Restringe una línea a ser perfectamente vertical
- **Horizontal**: Restringe una línea a ser perfectamente horizontal
- **Tangente**: Hace una línea tangente a un círculo o arco
- **Perpendicular**: Fuerza dos líneas, una línea y un arco/círculo, o dos arcos/círculos a encontrarse a 90 grados
- **Punto en Línea/Forma**: Restringe un punto a estar sobre una línea, arco o círculo
- **Simetría**: Crea relaciones simétricas entre elementos. Soporta dos modos:
  - **Simetría de Punto**: Selecciona 3 puntos (el primero es el centro)
  - **Simetría de Línea**: Selecciona 2 puntos y 1 línea (la línea es el eje)

### Restricciones Dimensionales

- **Distancia**: Establece la distancia exacta entre dos puntos o a lo largo de una línea
- **Diámetro**: Define el diámetro de un círculo
- **Radio**: Establece el radio de un círculo o arco
- **Ángulo**: Aplica un ángulo específico entre dos líneas
- **Relación de Aspecto**: Fuerza la proporción entre dos distancias a ser igual a un valor especificado
- **Igual Longitud/Radio**: Fuerza múltiples elementos (líneas, arcos o círculos) a tener la misma longitud o radio
- **Igual Distancia**: Fuerza la distancia entre dos pares de puntos a ser igual

## Interfaz de Menú Circular

El dibujador cuenta con un menú circular sensible al contexto que proporciona acceso rápido a todas las herramientas de dibujo y restricciones. Este menú radial aparece cuando haces clic derecho en el espacio de trabajo del boceto y se adapta basándose en tu contexto y selección actual.

Los elementos del menú circular muestran dinámicamente las opciones disponibles basándose en lo que tienes seleccionado. Por ejemplo, al hacer clic en espacio vacío, verás herramientas de dibujo. Al hacer clic en geometría seleccionada, verás restricciones aplicables.

![Menú Circular del Dibujador](/screenshots/sketcher-pie-menu.png)

## Atajos de Teclado

El dibujador proporciona atajos de teclado para un flujo de trabajo eficiente:

### Atajos de Herramientas
- `Espacio`: Herramienta de selección
- `G+L`: Herramienta de línea
- `G+A`: Herramienta de arco
- `G+C`: Herramienta de círculo
- `G+R`: Herramienta de rectángulo
- `G+O`: Herramienta de rectángulo redondeado
- `G+F`: Herramienta de relleno de área
- `G+T`: Herramienta de caja de texto
- `G+N`: Alternar modo construcción en la selección

### Atajos de Acción
- `C+H`: Añadir chaflán en esquina
- `C+F`: Añadir redondeo en esquina

### Atajos de Restricciones
- `H`: Aplicar restricción Horizontal
- `V`: Aplicar restricción Vertical
- `N`: Aplicar restricción Perpendicular
- `T`: Aplicar restricción Tangente
- `E`: Aplicar restricción Igual
- `O` o `C`: Aplicar restricción de Alineación (Coincidente)
- `S`: Aplicar restricción de Simetría
- `K+D`: Aplicar restricción de Distancia
- `K+R`: Aplicar restricción de Radio
- `K+O`: Aplicar restricción de Diámetro
- `K+A`: Aplicar restricción de Ángulo
- `K+X`: Aplicar restricción de Relación de Aspecto

### Atajos Generales
- `Ctrl+Z`: Deshacer
- `Ctrl+Y` o `Ctrl+Shift+Z`: Rehacer
- `Eliminar`: Eliminar elementos seleccionados
- `Escape`: Cancelar operación actual o deseleccionar
- `F`: Ajustar vista al contenido

## Modo Construcción

El modo construcción te permite marcar entidades como "geometría de construcción" - elementos auxiliares usados para guiar tu diseño pero que no son parte del resultado final. Las entidades de construcción se muestran de manera diferente (típicamente como líneas punteadas) y no se incluyen cuando el boceto se usa para corte láser o grabado.

Para alternar modo construcción:
- Selecciona una o más entidades
- Presiona `N` o `G+N`, o usa la opción Construcción en el menú circular

Las entidades de construcción son útiles para:
- Crear líneas y círculos de referencia
- Definir geometría temporal para alineación
- Construir formas complejas a partir de un marco de guías

## Chaflán y Redondeo

El dibujador proporciona herramientas para modificar esquinas de tu geometría:

- **Chaflán**: Reemplaza una esquina aguda con un borde biselado. Selecciona un punto de unión (donde dos líneas se encuentran) y aplica la acción de chaflán.
- **Redondeo**: Reemplaza una esquina aguda con un borde redondeado. Selecciona un punto de unión (donde dos líneas se encuentran) y aplica la acción de redondeo.

Para usar chaflán o redondeo:
1. Selecciona un punto de unión donde dos líneas se encuentran
2. Presiona `C+H` para chaflán o `C+F` para redondeo
3. Usa el menú circular o atajos de teclado para aplicar la modificación

## Importar y Exportar

### Exportar Objetos

Puedes exportar cualquier pieza de trabajo seleccionada a varios formatos vectoriales:

1. Selecciona una pieza de trabajo en el lienzo
2. Elige **Objeto → Exportar Objeto...** (o clic derecho y selecciona del menú contextual)
3. Elige el formato de exportación:
   - **RFS (.rfs)**: Formato de boceto paramétrico nativo de Rayforge - preserva todas las restricciones y puede reimportarse para edición
   - **SVG (.svg)**: Formato vectorial estándar - ampliamente compatible con software de diseño
   - **DXF (.dxf)**: Formato de intercambio CAD - compatible con la mayoría de aplicaciones CAD

### Guardar Bocetos

Puedes guardar tus bocetos 2D en archivos para reutilizarlos en otros proyectos. Todas las restricciones paramétricas se preservan al guardar, asegurando que tus diseños mantengan sus relaciones geométricas.

### Importar Bocetos

Los bocetos guardados pueden importarse a cualquier espacio de trabajo, permitiéndote crear una biblioteca de elementos de diseño comúnmente usados. El proceso de importación mantiene todas las restricciones y relaciones dimensionales.

## Consejos de Flujo de Trabajo

1. **Comienza con Geometría Básica**: Crea formas básicas primero, luego refina con restricciones
2. **Usa Restricciones Temprano**: Aplica restricciones mientras construyes para mantener la intención del diseño
3. **Verifica Estado de Restricciones**: El sistema indica cuándo los bocetos están completamente restringidos
4. **Observa Conflictos**: Las restricciones que entran en conflicto entre sí se resaltan en rojo
5. **Utiliza Simetría**: Las restricciones de simetría pueden acelerar significativamente diseños complejos
6. **Itera y Refina**: No dudes en modificar restricciones para lograr el resultado deseado

## Funciones de Edición

- **Soporte Completo de Deshacer/Rehacer**: Todo el estado del boceto se guarda con cada operación
- **Cursor Dinámico**: El cursor cambia para reflejar la herramienta de dibujo activa
- **Visualización de Restricciones**: Las restricciones aplicadas se indican claramente en la interfaz
- **Actualizaciones en Tiempo Real**: Los cambios en restricciones actualizan inmediatamente la geometría
- **Edición con Doble Clic**: Doble clic en restricciones dimensionales (Distancia, Radio, Diámetro, Ángulo, Relación de Aspecto) abre un diálogo para editar sus valores
- **Expresiones Paramétricas**: Las restricciones dimensionales soportan expresiones, permitiendo que los valores se calculen desde otros parámetros (ej., `ancho/2` para un radio que es la mitad del ancho)
