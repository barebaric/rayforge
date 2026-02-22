# Materiales

![Ajustes de Materiales](/screenshots/application-materials.png)

Las bibliotecas de materiales en Rayforge te permiten organizar y gestionar colecciones de materiales para tus proyectos de corte y grabado láser. Esta guía explica la diferencia entre bibliotecas principales y de usuario, y cómo crear tus propias bibliotecas y añadir materiales a ellas.

:::note
Las bibliotecas de materiales actualmente no tienen uso funcional - asignar un material
meramente afecta la apariencia estética. En futuras versiones, los materiales
se usarán para derivar parámetros funcionales.
:::


## Creando una Nueva Biblioteca

Para crear tu propia biblioteca de materiales:

1. Abre el menú **Configuración** y selecciona **Materiales**
2. Haz clic en el botón **Añadir Nueva Biblioteca** para crear una nueva biblioteca
3. Ingresa un nombre descriptivo para tu biblioteca (ej., "Materiales de Mi Taller")
4. Haz clic en **Crear** para finalizar

Tu nueva biblioteca será creada en el directorio de datos de usuario y estará disponible inmediatamente.


## Añadiendo Materiales a las Bibliotecas

### Creando un Nuevo Material

1. Selecciona la biblioteca donde quieres añadir el material
2. Haz clic en el botón **Añadir Nuevo Material** en la lista de materiales
3. Completa las propiedades del material:
   - **Nombre**: Nombre legible para humanos
   - **Categoría**: Categoría de agrupación (ej., "Madera", "Acrílico")
   - **Apariencia**: Propiedades visuales (color)
4. Haz clic en **Guardar** para añadir el material a la biblioteca

### Propiedades de Material Explicadas

#### Nombre
- Nombre legible para humanos mostrado en la interfaz
- Puede contener espacios y caracteres especiales

#### Categoría
- Usada para organizar materiales dentro de la biblioteca
- Categorías comunes incluyen: Madera, Acrílico, Metal, Papel, Cuero
- Puedes crear categorías personalizadas según sea necesario

#### Color

El color solo se usa para la apariencia visual en la superficie de trabajo - no
afecta la trayectoria del láser de ninguna manera.


## Gestionando Materiales Existentes

### Editando Materiales

1. Selecciona el material que quieres editar
2. Haz clic en el botón **Editar**
3. Modifica las propiedades deseadas
4. Haz clic en **Guardar** para aplicar los cambios

### Eliminando Materiales

1. Selecciona el material que quieres eliminar
2. Haz clic en el botón **Eliminar**
3. Confirma la eliminación en el diálogo

:::warning
Eliminar un material es permanente y no se puede deshacer.
:::
