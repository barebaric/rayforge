# Recetas y Ajustes

![Ajustes de Recetas](/screenshots/app-settings-recipes.webp)

Rayforge proporciona un potente sistema de recetas que te permite crear, gestionar y aplicar ajustes
consistentes en tus proyectos de corte láser. Esta guía cubre el viaje completo del usuario desde
crear recetas en los ajustes generales hasta aplicarlas a operaciones y gestionar ajustes a nivel de
paso.

## Resumen

El sistema de recetas consiste en tres componentes principales:

1. **Gestión de Recetas**: Crear y gestionar preajustes de ajustes reutilizables
2. **Gestión de Material en Stock**: Definir propiedades de material y grosor
3. **Ajustes de Paso**: Aplicar y afinar ajustes para operaciones individuales

## Gestión de Recetas

### Creando Recetas

Las recetas son preajustes nombrados que contienen todos los ajustes necesarios para operaciones
específicas. Puedes crear recetas a través de la interfaz de ajustes principales:

#### 1. Acceder al Gestor de Recetas

Menú: Editar → Ajustes, luego selecciona Recetas

#### 2. Crear Nueva Receta

Haz clic en "Añadir Nueva Receta" para abrir el diálogo del editor de recetas.

**Pestaña General** - Establece el nombre y descripción de la receta:

![Editor de Recetas - Pestaña General](/screenshots/recipe-editor-general.webp)

Completa la información básica:

- **Nombre**: Nombre descriptivo (ej., "Corte Contrachapado 3mm")
- **Descripción**: Descripción detallada opcional

#### 3. Definir Criterios de Aplicabilidad

**Pestaña de Aplicabilidad** - Define cuándo se debe sugerir esta receta:

![Editor de Recetas - Pestaña Aplicabilidad](/screenshots/recipe-editor-applicability.webp)

Todos los criterios son opcionales - deja cualquier campo en su valor "Any" para coincidir con todo:

- **Máquina**: Elige una máquina específica o déjala como "Cualquier Máquina"
- **Tipo de Tarea**: Selecciona la categoría de operación a la que se aplica esta receta (Corte,
  Grabado, etc.), o déjala como "Any" para aplicarla a todos los tipos de tarea
- **Tipo de Paso**: Restringe la receta a un tipo de operación específico (ej., "Contorno" o
  "Raster"). La lista se filtra a los tipos de paso que soportan el tipo de tarea seleccionado.
  Déjalo como "Any Type" para coincidir con cada tipo de paso dentro de la tarea
- **Material**: Selecciona un tipo de material o déjalo abierto para cualquier material
- **Grosor Mín/Máx**: Establece valores de grosor de stock mínimo y máximo

#### 4. Configurar Ajustes

**Pestaña de Ajustes** - Ajusta potencia, velocidad y otros parámetros. Cuando la receta se dirige a
un **tipo de paso** específico, el editor muestra dos páginas de ajustes: una página "Laser" con los
ajustes de proceso compartidos (potencia, asistencia de aire, etc.) y una página "Ajustes de Paso"
con los atributos específicos de ese tipo de paso (ej., lado de corte, orden de corte):

![Editor de Recetas - Pestaña Laser](/screenshots/recipe-editor-laser.webp)

![Editor de Recetas - Pestaña Ajustes de Paso](/screenshots/recipe-editor-step-settings.webp)

- Seleccionar solo un **tipo de tarea** (con "Any Type" como tipo de paso) muestra una única página
  "Ajustes" con los ajustes de proceso para esa tarea
- Dejar ambos en "Any" muestra solo los ajustes de movimiento base (velocidad de corte y velocidad
  de viaje) que comparten todos los pasos

Cada fila de ajustes tiene un conmutador de aplicación (un botón de verificación junto a la fila):

- **Activado**: la receta aplica este ajuste al paso cuando se aplica
- **Desactivado**: la receta deja este ajuste en el paso sin cambios

**Pestaña de Post-Procesamiento** - Almacena ajustes de post-procesamiento (entrada/salida,
multipase, overscan y otros transformadores) en la receta para que se apliquen a los pasos que esta
tiene como objetivo:

![Editor de Recetas - Pestaña Post-Procesamiento](/screenshots/recipe-editor-post-processing.webp)

Cada transformador tiene un conmutador de aplicación (un botón de verificación junto a la fila):

- **Activado**: la receta aplica los parámetros del transformador al paso (su propio interruptor de
  habilitación decide si se activa o desactiva)
- **Desactivado**: la receta no toca este transformador al aplicarse

Cuando la receta tiene como objetivo varios tipos de paso, solo se muestran los transformadores
comunes a todos ellos.

### Sistema de Coincidencia de Recetas

Rayforge sugiere automáticamente y aplica las recetas más apropiadas basándose en:

- **Compatibilidad de máquina**: Las recetas pueden ser específicas de máquina
- **Compatibilidad de cabezal láser**: Las recetas pueden forzar un cabezal específico en la máquina
- **Coincidencia de material**: Las recetas pueden dirigirse a materiales específicos
- **Rangos de grosor**: Las recetas se aplican dentro de límites de grosor definidos
- **Coincidencia de tipo de tarea**: Las recetas están vinculadas a categorías de operación
  específicas
- **Coincidencia de tipo de paso**: Las recetas pueden dirigirse a un tipo de operación específico
  (ej., solo pasos "Contorno")

Una receta solo coincide cuando todos sus criterios se cumplen. Cuando se crea un nuevo paso,
Rayforge busca en la biblioteca de recetas las recetas coincidentes y aplica automáticamente la
mejor. El sistema usa un algoritmo de puntuación de especificidad para priorizar las recetas más
relevantes:

1. Las recetas específicas de máquina tienen mayor rango que las genéricas
2. Las recetas específicas de cabezal láser tienen mayor rango
3. Las recetas específicas de material tienen mayor rango
4. Las recetas específicas de grosor tienen mayor rango
5. Las recetas específicas de tipo de paso tienen mayor rango

### Aplicando Recetas a Pasos

Las recetas se aplican por paso. Abre los ajustes de cualquier paso y encuentra la fila "Receta" en
la sección "General":

- **Elegir...**: Abre una lista filtrable de recetas. Usa el campo de búsqueda o el interruptor
  "Mostrar solo recetas compatibles" para reducir la lista; las recetas compatibles coinciden con el
  tipo de tarea del paso, el tipo de paso, la máquina y los materiales en stock del documento.
  Seleccionar una receta aplica todos sus ajustes al paso.
- **Guardar Como...**: Abre el editor de recetas pre-rellenado con los ajustes, la máquina, el
  material y el grosor actuales del paso. Guardar la nueva receta la aplica al paso inmediatamente.
- **Actualizar**: Aparece cuando los ajustes del paso han divergido de la receta que se le aplicó
  (ej., después de cambiar un valor manualmente). Al hacer clic, sobrescribe la receta guardada con
  los ajustes actuales del paso.

El nombre de la receta aplicada actualmente se muestra en la fila. Los pasos sin una receta aplicada
están etiquetados como "Ajustes Manuales".

---

**Temas Relacionados**:

- [Materiales](materials) - Gestionando propiedades de materiales
- [Reglas de color](color-rules) - Asigna los colores SVG a tipos de paso al importar
- [Manejo de Stock](../features/stock-handling.md) - Trabajando con materiales en stock
- [Configuración de Máquina](../machine/general.md) - Configurando máquinas y cabezales láser
- [Resumen de Operaciones](../features/operations/contour.md) - Entendiendo diferentes tipos de
  operaciones
