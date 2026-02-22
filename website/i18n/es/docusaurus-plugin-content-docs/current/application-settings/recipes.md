# Recetas y Ajustes

![Ajustes de Recetas](/screenshots/application-recipes.png)

Rayforge proporciona un potente sistema de recetas que te permite crear, gestionar y aplicar ajustes consistentes en tus proyectos de corte láser. Esta guía cubre el viaje completo del usuario desde crear recetas en los ajustes generales hasta aplicarlas a operaciones y gestionar ajustes a nivel de paso.

## Resumen

El sistema de recetas consiste en tres componentes principales:

1. **Gestión de Recetas**: Crear y gestionar preajustes de ajustes reutilizables
2. **Gestión de Material en Stock**: Definir propiedades de material y grosor
3. **Ajustes de Paso**: Aplicar y afinar ajustes para operaciones individuales

## Gestión de Recetas

### Creando Recetas

Las recetas son preajustes nombrados que contienen todos los ajustes necesarios para operaciones específicas.
Puedes crear recetas a través de la interfaz de ajustes principales:

#### 1. Acceder al Gestor de Recetas

Menú: Editar → Preferencias → Recetas

#### 2. Crear Nueva Receta

Haz clic en "Añadir Nueva Receta" para abrir el diálogo del editor de recetas.

**Pestaña General** - Establece el nombre y descripción de la receta:

![Editor de Recetas - Pestaña General](/screenshots/recipe-editor-general.png)

Completa la información básica:

- **Nombre**: Nombre descriptivo (ej., "Corte Contrachapado 3mm")
- **Descripción**: Descripción detallada opcional

#### 3. Definir Criterios de Aplicabilidad

**Pestaña de Aplicabilidad** - Define cuándo se debe sugerir esta receta:

![Editor de Recetas - Pestaña Aplicabilidad](/screenshots/recipe-editor-applicability.png)

- **Tipo de Tarea**: Selecciona el tipo de operación (Corte, Grabado, etc.)
- **Máquina**: Elige una máquina específica o déjala como "Cualquier Máquina"
- **Material**: Selecciona un tipo de material o déjalo abierto para cualquier material
- **Rango de Grosor**: Establece valores de grosor mínimo y máximo

#### 4. Configurar Ajustes

**Pestaña de Ajustes** - Ajusta potencia, velocidad y otros parámetros:

![Editor de Recetas - Pestaña Ajustes](/screenshots/recipe-editor-settings.png)

- Ajusta potencia, velocidad y otros parámetros
- Los ajustes se adaptan automáticamente según el tipo de tarea seleccionado

### Sistema de Coincidencia de Recetas

Rayforge sugiere automáticamente las recetas más apropiadas basándose en:

- **Compatibilidad de máquina**: Las recetas pueden ser específicas de máquina
- **Coincidencia de material**: Las recetas pueden dirigirse a materiales específicos
- **Rangos de grosor**: Las recetas se aplican dentro de límites de grosor definidos
- **Coincidencia de capacidad**: Las recetas están vinculadas a tipos de operación específicos

El sistema usa un algoritmo de puntuación de especificidad para priorizar las recetas más relevantes:

1. Las recetas específicas de máquina tienen mayor rango que las genéricas
2. Las recetas específicas de cabezal láser tienen mayor rango
3. Las recetas específicas de material tienen mayor rango
4. Las recetas específicas de grosor tienen mayor rango

---

**Temas Relacionados**:

- [Materiales](materials) - Gestionando propiedades de materiales
- [Manejo de Stock](../features/stock-handling) - Trabajando con materiales en stock
- [Configuración de Máquina](../machine/general) - Configurando máquinas y cabezales láser
- [Resumen de Operaciones](../features/operations/contour) - Entendiendo diferentes tipos de operaciones
