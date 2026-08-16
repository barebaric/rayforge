# Manejo de Material Base

El material base en Rayforge representa el material físico que cortarás o grabarás. El material base es un concepto **global del documento**—tu documento puede tener uno o más elementos de material base, y existen independientemente de las capas.

## Añadir Material Base

El material base representa la pieza física de material con la que trabajarás. Para añadir material base a tu documento:

1. Abre la pestaña de **Activos** en el panel inferior
2. Haz clic en **Añadir Material Base** — este botón se muestra cuando la lista de activos está vacía
3. Se creará un nuevo elemento de material base con dimensiones por defecto (80% del área de trabajo de tu máquina), colocado con una esquina en el origen de la máquina
4. El material base aparecerá como un rectángulo en el área de trabajo

Alternativamente, puedes añadir material base mediante:

- El botón **+** en la pestaña de Activos y eligiendo **Añadir Material Base**
- Clic derecho en un área vacía de la lista de activos y seleccionando **Nuevo Material Base**
- Clic derecho en un área vacía del lienzo y seleccionando **Nuevo Material Base**
- Pulsando `Ctrl+Alt+S`

También puedes arrastrar un activo de material base desde la pestaña de Activos al lienzo para colocar otro elemento de material base.

### Propiedades del Material Base

Haz doble clic en un activo de material base en la pestaña de Activos para abrir el diálogo **Propiedades de Material Base**. Cada elemento de material base tiene las siguientes propiedades:

- **Nombre:** Un nombre descriptivo para identificación (autonumerado como "Material Base 1", "Material Base 2", etc.)
- **Espesor:** El espesor del material (opcional pero recomendado para previsualización 3D precisa)
- **Material:** El tipo de material (asignado en el siguiente paso)

Las dimensiones del material base se ajustan seleccionando el material base en el área de trabajo y arrastrando sus manijas de las esquinas.

### Gestión de Elementos de Material Base

- **Renombrar:** Abre el diálogo de Propiedades de Material Base (doble clic en el activo) y edita el campo de nombre
- **Redimensionar:** Selecciona el elemento de material base en el área de trabajo y arrastra las manijas de las esquinas para redimensionar
- **Mover:** Selecciona el elemento de material base en el área de trabajo y arrastra para reposicionar
- **Editar propiedades:** Haz doble clic en el activo de material base en la pestaña de Activos
- **Duplicar:** Haz clic derecho en el activo de material base en la pestaña de Activos y selecciona **Duplicar**
- **Eliminar:** Haz clic derecho en el activo de material base en la pestaña de Activos y selecciona **Eliminar**, o selecciona el activo y pulsa `Supr`

## Asignar Material

Una vez que tienes material base definido, puedes asignarle un material:

1. Haz doble clic en el activo de material base en la pestaña de **Activos** para abrir el diálogo de Propiedades de Material Base
2. En el diálogo de Propiedades de Material Base, haz clic en el botón **Seleccionar** junto al campo de Material
3. Navega por tus bibliotecas de materiales y selecciona el material apropiado
4. El material base se actualizará para mostrar la apariencia visual del material

### Propiedades del Material

Los materiales definen las propiedades visuales de tu material base:

- **Apariencia visual:** Color y patrón para visualización
- **Categoría:** Agrupación (ej., "Madera", "Acrílico", "Metal")
- **Descripción:** Información adicional sobre el material

Nota: Las propiedades del material se definen en bibliotecas de materiales y no pueden editarse a través del diálogo de propiedades del material base. Las propiedades del material base solo te permiten asignar un material a un elemento de material base.

## Convertir Piezas en Material Base

Puedes convertir cualquier pieza en un elemento de material base. Esto es útil cuando tienes una pieza de material con forma irregular y quieres usar su contorno exacto como límite del material base.

Para convertir una pieza en material base:

1. Haz clic derecho en la pieza en el lienzo
2. Selecciona **Convertir en Material Base** del menú contextual
3. La pieza será reemplazada por un nuevo elemento de material base con la misma forma y posición

El nuevo elemento de material base:

- Usa la geometría de la pieza como su límite
- Hereda el nombre de la pieza
- Puede tener un material asignado como cualquier otro elemento de material base

## Auto-Layout

La función de auto-layout te ayuda a organizar eficientemente tus elementos de diseño dentro de los límites del material base:

1. Selecciona los elementos que quieres organizar (o no selecciones nada para organizar todos los elementos en la capa activa)
2. Haz clic en el botón **Organizar** en la barra de herramientas y selecciona **Auto Layout (empaquetar piezas)**
3. Rayforge organizará automáticamente los elementos para optimizar el uso del material

### Comportamiento del Auto-Layout

El algoritmo de auto-layout organiza elementos dentro de los elementos de material base visibles en tu documento:

- **Si elementos de material base están definidos:** Los elementos se organizan dentro de los límites de los elementos de material base visibles
- **Si ningún material base está definido:** Los elementos se organizan en toda el área de trabajo de la máquina

El algoritmo considera:

- **Límites de elementos:** Respeta las dimensiones de cada elemento de diseño
- **Rotación:** Puede rotar elementos en incrementos de 90 grados para mejor ajuste
- **Espaciado:** Mantiene un margen entre elementos (por defecto 0.5mm)
- **Límites del material base:** Mantiene todos los elementos dentro de los límites definidos

### Alternativas de Layout Manual

Si prefieres más control, Rayforge también ofrece herramientas de layout manual:

- **Herramientas de alineación:** Alinear izquierda, derecha, centro, arriba, abajo
- **Herramientas de distribución:** Distribuir elementos horizontal o verticalmente
- **Posicionamiento individual:** Haz clic y arrastra elementos para colocarlos manualmente

## Consejos para un Manejo Efectivo del Material Base

1. **Comienza con dimensiones precisas del material base** - Mide tu material con precisión para mejores resultados
2. **Usa nombres descriptivos** - Nombra tus elementos de material base claramente (ej., "Contrachapado Abedul 3mm")
3. **Configura el espesor del material** - Esto puede ser útil para cálculos futuros y referencia
4. **Asigna materiales temprano** - Esto asegura representación visual correcta desde el inicio
5. **Usa material base irregular para retales** - Convierte piezas en material base cuando uses material sobrante con formas personalizadas
6. **Verifica el ajuste antes de cortar** - Usa la vista 2D para verificar que todo cabe en tu material base

## Solución de Problemas

### El auto-layout no funciona como se espera

- Asegúrate de que al menos un elemento de material base esté definido (el auto-layout usa el primer elemento de material base visible como límite)
- Asegúrate que los elementos no estén agrupados (desagrupa primero)
- Intenta reducir el número de elementos seleccionados a la vez
- Verifica que los elementos caben dentro de los límites del material base
