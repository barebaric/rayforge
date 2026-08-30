# Reglas de color

Las reglas de color te permiten asignar un tipo de paso a un color específico para que se elija
automáticamente la operación correcta al importar un SVG, PDF u otro archivo vectorial. En lugar de
crear pasos manualmente para cada capa importada, Rayforge lee el color de cada forma y aplica la
regla que coincida.

## Cómo funciona

Cuando importas un archivo vectorial, Rayforge puede agrupar las formas entrantes por su color. Cada
color distinto se convierte en una capa. Si existe una regla de color para ese color, la capa recibe
automáticamente el tipo de paso de la regla. Los colores sin una regla reciben el comportamiento
predeterminado (Contorno para los contornos, más Grabado si las formas tienen rellenos).

Después de asignar el tipo de paso, se ejecuta encima el sistema normal de
[coincidencia de recetas](recipes) — así que las reglas de color determinan _qué_ operación se
ejecuta, y las recetas determinan _cómo_ se ejecuta (potencia, velocidad, pasadas, etc.).

## Crear reglas de color

### 1. Abrir la página de Reglas de color

Menú: **Editar → Configuración** y luego selecciona **Reglas de color** en la barra lateral.

### 2. Añadir una regla

Haz clic en **Añadir regla de color** para abrir el diálogo del editor:

- **Color** — Elige el color SVG que debe activar esta regla. Usa el selector de color para
  coincidir con el color del trazo o del relleno de tu software de diseño.
- **Etiqueta** _(opcional)_ — Un nombre descriptivo que se muestra en la lista de reglas (p. ej.
  "Cortar rojo", "Grabar azul"). Si se deja en blanco, se usa el valor hexadecimal.
- **Tipo de paso** — La operación que se creará al importar este color. Está disponible cualquier
  tipo de paso registrado, incluidos los proporcionados por [complementos](addons) (p. ej. Shrink
  Wrap, Material Test Grid).

### 3. Guardar

Haz clic en **Añadir** para guardar la regla. Surte efecto de inmediato en la siguiente importación.
Las reglas se guardan en tu configuración de usuario y persisten entre sesiones.

:::tip Coincidir colores exactamente Las reglas de color coinciden por valor hexadecimal exacto. Al
elegir un color en tu software de diseño (Inkscape, Illustrator, etc.), anota el código hexadecimal
exacto e introduce el mismo valor en Rayforge. Por ejemplo, `#e34c4c` en tu SVG debe ser `#e34c4c`
en la regla — incluso una diferencia de un dígito impedirá la coincidencia. :::

## Administrar reglas

Cada regla de la lista muestra una muestra de color, la etiqueta, el tipo de paso y los botones de
editar/eliminar.

- **Editar** — Cambia el color, la etiqueta o el tipo de paso. Cambiar el color de una regla
  existente la reemplaza (el color anterior se elimina).
- **Eliminar** — Elimina la regla de forma permanente.
- **Tipos de paso no disponibles** — Si se ha desinstalado el complemento del tipo de paso, aparece
  un icono de advertencia junto a la regla. La regla se conserva para que puedas arreglarla o
  reinstalar el complemento. Durante la importación, las capas que coinciden con una regla con un
  tipo de paso no disponible recurren al comportamiento predeterminado.

## Comportamiento de importación

### Agrupación automática por color

Cuando existen reglas de color, el diálogo de importación cambia automáticamente a **Colores** como
origen de capas para los archivos que contienen colores distintos. Esto garantiza que cada color se
convierta en su propia capa para que puedan aplicarse las reglas. Todavía puedes volver a **Capas
SVG** u otros orígenes en el diálogo si lo prefieres.

### Qué activa una regla

Una regla de color se aplica cuando:

1. El archivo se importa con **Colores** como origen de capas.
2. El color del trazo o del relleno de una forma coincide exactamente con el color de la regla.
3. El tipo de paso de la regla está registrado actualmente.

Las reglas **no** se aplican a los archivos importados con los orígenes de capas **Capas SVG** o
**Aplanar**, porque esos orígenes no agrupan por color.

## Flujo de trabajo de ejemplo

Una configuración habitual para diseños SVG de varios colores:

1. **En tu software de diseño**, asigna colores distintos a diferentes operaciones:
   - Rojo (`#ff0000`) para contornos de corte
   - Azul (`#0000ff`) para grabado
   - Verde (`#00ff00`) para marcado

2. **En Rayforge**, crea tres reglas de color:
   - `#ff0000` → Contorno
   - `#0000ff` → Grabado
   - `#00ff00` → Contorno (con ajustes de receta diferentes)

3. **Importa el SVG.** El diálogo de importación selecciona automáticamente Colores, y cada grupo de
   colores recibe su tipo de paso automáticamente.

4. **Ajusta finamente** con [recetas](recipes) para establecer la potencia, la velocidad y otros
   parámetros por tipo de paso.

## Reglas de color y recetas

Las reglas de color y las recetas son complementarias:

| Característica  | Lo que establece                  | Cuándo se aplica |
| --------------- | --------------------------------- | ---------------- |
| Reglas de color | Tipo de paso (Contorno, etc.)     | Al importar      |
| Recetas         | Ajustes del paso (potencia, etc.) | Al crear el paso |

Una configuración típica es usar las reglas de color para elegir la operación y las recetas para
configurar los parámetros. Por ejemplo, una regla de color rojo se asigna a Contorno, y una receta
limitada al tipo de paso Contorno en tu material actual aplica la velocidad y la potencia de corte
adecuadas.

---

**Temas relacionados**:

- [Recetas](recipes) - Aplicar ajustes preestablecidos de potencia, velocidad y parámetros
- [Importar archivos](../files/importing.md) - Opciones de importación de SVG y vectores
- [Flujo de trabajo multicapa](../features/multi-layer.md) - Organización de capas
- [Operaciones](../features/operations/contour.md) - Referencia de tipos de paso
