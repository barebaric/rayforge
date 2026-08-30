# Materiales

![Ajustes de Materiales](/screenshots/app-settings-materials.webp)

Las bibliotecas de materiales en Rayforge te permiten organizar y gestionar colecciones de materiales para tus proyectos de corte y grabado láser. Esta guía explica la diferencia entre bibliotecas principales y de usuario, y cómo crear tus propias bibliotecas y añadir materiales a ellas.

:::note
Asignar un material a un elemento de material de base afecta tanto su
apariencia visual en el lienzo 2D y 3D como qué [recetas](recipes.md)
se le aplican: las recetas específicas de material coinciden con el
material asignado. En futuras versiones, los materiales se usarán para
derivar más parámetros funcionales.
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
   - **Apariencia**: Propiedades visuales (ver abajo)
4. Haz clic en **Guardar** para añadir el material a la biblioteca

### Propiedades de Material Explicadas

#### Nombre

- Nombre legible para humanos mostrado en la interfaz
- Puede contener espacios y caracteres especiales

#### Categoría

- Usada para organizar materiales dentro de la biblioteca
- Categorías comunes incluyen: Madera, Acrílico, Metal, Papel, Cuero
- Puedes crear categorías personalizadas según sea necesario

#### Textura

Una imagen de textura (WebP o PNG) que se repite en mosaico sobre la
superficie del material. Cuando se establece, el material se renderiza con
la textura en lugar de un color plano. Las texturas se pueden optimizar a
WebP con el script `scripts/optimize_material_textures.py` para mantener
los archivos de material pequeños.

#### Escala de textura

El tamaño (en mm) que una tesela de textura cubre sobre el material.
Valores más pequeños repiten la textura más a menudo sobre la misma
superficie.

#### Color

Un color de tinte opcional. Cuando se establece, la textura del material
se tiñe con este color; cuando no, la textura se muestra tal cual. Esto
permite que un único material texturizado (ej., "Acrílico") cubra
múltiples variantes de color: el color se aplica por elemento de material
de base en el diálogo [Propiedades del material de
base](../features/stock-handling.md). El color solo se usa para la
apariencia visual en la superficie de trabajo - no afecta la trayectoria
del láser de ninguna manera.

#### Rugosidad

Un valor de 0-1 que describe cuán rugosa o pulida aparece la superficie
en la vista 3D. Valores más bajos se ven brillantes, valores más altos se
ven mate.

#### Metálico

Un valor de 0-1 que describe si la superficie refleja la luz como un
metal en la vista 3D. Establece 1 para materiales metálicos, 0 para no
metálicos.

#### Absorción

:::note Nuevo en 1.11
Los datos de absorción impulsan el [modelo de quemado físico](../ui/3d-preview.md#physical-burn-model)
en la vista previa 3D.
:::

Coeficientes de absorción por longitud de onda (0–1) describen cuánta
energía del láser absorbe un material a una longitud de onda determinada.
La vista previa 3D los usa, junto con la longitud de onda, la potencia
óptica y el tamaño del punto de tu cabezal láser, para calcular la
fluencia (J/cm²) entregada y renderizar un efecto de quemado físicamente
motivado sobre el material de base.

Añade un bloque `absorption` bajo `appearance` en el YAML del material:

```yaml
appearance:
  absorption:
    blue: 0.7 # ~445 nm láseres de diodo
    ir: 0.25 # ~1064 nm láseres de fibra / IR
    co2: 0.9 # ~10600 nm láseres CO2
  # ...otras propiedades de apariencia
```

| Banda  | Longitud de onda representativa | Láseres típicos       |
| ------ | ------------------------------- | --------------------- |
| `blue` | 445 nm                          | Láseres de diodo azul |
| `ir`   | 1064 nm                         | Láseres de fibra      |
| `co2`  | 10600 nm                        | Láseres de tubo CO2   |

Cuando falta una banda, se usa un valor predeterminado conservador. La
biblioteca de materiales incluida envía valores de absorción investigados
para todos los materiales incluidos; el modelo de quemado aún no está
completamente calibrado, por lo que se agradecen contribuciones de datos
de prueba del mundo real.

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
