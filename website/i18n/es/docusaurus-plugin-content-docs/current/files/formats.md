# Formatos de archivo compatibles

Esta página proporciona información detallada sobre todos los formatos de
archivo compatibles con Rayforge, incluyendo capacidades, limitaciones y
recomendaciones.

## Resumen de formatos

### Referencia rápida

| Formato              | Tipo     | Importar           | Exportar        | Uso recomendado                |
| -------------------- | -------- | ------------------ | --------------- | ------------------------------ |
| **SVG**              | Vector   | ✓ Directo / Traza  | ✓ Exportar obj. | Formato de diseño principal    |
| **DXF**              | Vector   | ✓ Directo          | ✓ Exportar obj. | Intercambio CAD                |
| **PDF**              | Mixto    | ✓ Directo / Traza  | –               | Documentos con contenido vec.  |
| **PNG**              | Raster   | ✓ Traza            | –               | Fotos, imágenes                |
| **JPEG**             | Raster   | ✓ Traza            | –               | Fotos                          |
| **BMP**              | Raster   | ✓ Traza            | –               | Gráficos simples               |
| **RFS**              | Bosquejo | ✓ Directo          | ✓ Exportar obj. | Bosquejos paramétricos         |
| **G-code**           | Control  | –                  | ✓ Principal     | Salida de máquina              |
| **Proyecto Rayforge**| Proyecto | ✓                  | ✓               | Guardar/cargar proyectos       |

---

## Formatos vectoriales

### SVG (Scalable Vector Graphics)

**Extensión:** `.svg`
**Tipo MIME:** `image/svg+xml`
**Importación:** Análisis vectorial directo o traza de mapa de bits
**Exportación:** Exportar objeto (solo geometría)

**¿Qué es SVG?**

SVG es un formato de imagen vectorial basado en XML. Es el **formato preferido**
para importar diseños en Rayforge.

**Características compatibles:**

- ✓ Trazados (líneas, curvas, arcos)
- ✓ Formas básicas (rectángulos, círculos, elipses, polígonos)
- ✓ Grupos y transformaciones
- ✓ Colores de trazo y relleno
- ✓ Múltiples capas
- ✓ Transformaciones de coordenadas (trasladar, rotar, escalar)

**Características no compatibles/limitadas:**

- ✗ Texto (debe convertirse a trazados primero)
- ✗ Degradados (simplificados o ignorados)
- ✗ Filtros y efectos (ignorados)
- ✗ Máscaras y trazados de recorte (pueden no funcionar correctamente)
- ✗ Imágenes ráster integradas (importadas por separado si es posible)
- ✗ Estilos de trazo complejos (los guiones pueden simplificarse)
- ✗ Símbolos y elementos use (las instancias pueden no actualizarse)

**Notas de exportación:**

Al exportar una pieza de trabajo a SVG, Rayforge exporta la geometría como
trazados vectoriales con:

- Representación solo trazo (sin relleno)
- Unidades en milímetros
- Color de trazo negro

**Mejores prácticas:**

1. **Use el formato SVG simple** (no Inkscape SVG ni otras variantes
   específicas de herramientas)
2. **Convierta el texto a trazados** antes de exportar
3. **Simplifique los trazados complejos** para reducir la cantidad de nodos
4. **Aplane los grupos** cuando sea posible
5. **Elimine elementos no utilizados** (guías, cuadrículas, capas ocultas)
6. **Configure las unidades del documento** en mm (unidad nativa de Rayforge)

**Recomendaciones de software:**

- **Inkscape** (gratuito) - Excelente soporte SVG, formato nativo

---

### DXF (Drawing Exchange Format)

**Extensión:** `.dxf`
**Tipo MIME:** `application/dxf`, `image/vnd.dxf`
**Importación:** Análisis vectorial directo
**Exportación:** Exportar objeto (solo geometría)

**¿Qué es DXF?**

DXF es un formato de dibujo de AutoCAD, ampliamente utilizado para el
intercambio de datos CAD.

**Versiones compatibles:**

- ✓ **R12/LT2** (recomendada - mejor compatibilidad)
- ✓ R13, R14
- ✓ R2000 y posteriores (generalmente funciona, pero R12 es más segura)

**Entidades compatibles:**

- ✓ Líneas (LINE)
- ✓ Polilíneas (LWPOLYLINE, POLYLINE)
- ✓ Arcos (ARC)
- ✓ Círculos (CIRCLE)
- ✓ Splines (SPLINE) - convertidos a polilíneas
- ✓ Elipses (ELLIPSE)
- ✓ Capas

**Características no compatibles/limitadas:**

- ✗ Entidades 3D (use proyección 2D)
- ✗ Acotaciones y anotaciones (ignoradas)
- ✗ Bloques/inserciones (pueden no instanciar correctamente)
- ✗ Tipos de línea complejos (simplificados a línea continua)
- ✗ Texto (ignorado, convierta a contornos primero)
- ✗ Tramados (pueden simplificarse o ignorarse)

**Notas de exportación:**

Al exportar una pieza de trabajo a DXF, Rayforge exporta:

- Líneas como entidades LWPOLYLINE
- Arcos como entidades ARC
- Curvas Bézier como entidades SPLINE
- Unidades en milímetros (INSUNITS = 4)

---

### RFS (Bosquejo de Rayforge)

**Extensión:** `.rfs`
**Tipo MIME:** `application/x-rayforge-sketch`
**Importación:** Directo (piezas basadas en bosquejos)
**Exportación:** Exportar objeto (piezas basadas en bosquejos)

**¿Qué es RFS?**

RFS es el formato nativo de bosquejo paramétrico de Rayforge. Preserva todos
los elementos geométricos y restricciones paramétricas, permitiendo guardar y
compartir bosquejos completamente editables.

**Características:**

- ✓ Todos los elementos geométricos (líneas, arcos, círculos, rectángulos,
  etc.)
- ✓ Todas las restricciones paramétricas
- ✓ Valores dimensionales y expresiones
- ✓ Áreas de relleno

**Cuándo usar:**

- Guardar diseños paramétricos reutilizables
- Compartir bosquejos editables con otros usuarios de Rayforge
- Archivar trabajos en progreso

---

### PDF (Portable Document Format)

**Extensión:** `.pdf`
**Tipo MIME:** `application/pdf`
**Importación:** Vectores directos (con soporte de capas) o renderizar y trazar
**Exportación:** No compatible

**¿Qué es la importación PDF?**

Los archivos PDF pueden contener trazados vectoriales reales, y Rayforge los
importa directamente cuando están disponibles — proporcionando la misma
geometría limpia que obtendría de un SVG. Si el PDF tiene capas, cada capa
puede importarse como una pieza de trabajo separada.

Para PDFs sin contenido vectorial utilizable (documentos escaneados, fotos),
Rayforge recurre al renderizado y trazado.

**Capacidades:**

- ✓ **Importación vectorial directa** para PDFs basados en vectores
- ✓ **Detección y selección de capas** — elija qué capas importar
- ✓ Renderizado y trazado alternativo para contenido ráster

**Limitaciones:**

- Solo la primera página — los PDFs de varias páginas importan la página 1
- El texto puede necesitar convertirse a contornos en la aplicación de origen

**Cuándo usar:**

- PDFs recibidos de diseñadores que contienen arte vectorial
- Cualquier PDF con capas bien organizadas
- Cuando SVG o DXF no están disponibles desde la fuente

---

## Formatos ráster

Todos los formatos ráster se **importan mediante trazado** — se convierten
automáticamente a trazados vectoriales.

### PNG (Portable Network Graphics)

**Extensión:** `.png`
**Tipo MIME:** `image/png`
**Importación:** Traza a vectores
**Exportación:** No compatible

**Características:**

- **Compresión sin pérdida** - Sin pérdida de calidad
- **Soporte de transparencia** - Canal alfa preservado
- **Adecuado para:** Logotipos, arte lineal, capturas de pantalla, cualquier
  elemento que necesite transparencia

**Calidad de trazado:** (Excelente para imágenes de alto contraste)

**Mejores prácticas:**

- Use PNG para logotipos y gráficos con bordes nítidos
- Asegure alto contraste entre el primer plano y el fondo
- El fondo transparente funciona mejor que el blanco

---

### JPEG (Joint Photographic Experts Group)

**Extensión:** `.jpg`, `.jpeg`
**Tipo MIME:** `image/jpeg`
**Importación:** Traza a vectores
**Exportación:** No compatible

**Características:**

- **Compresión con pérdida** - Alguna pérdida de calidad
- **Sin transparencia** - Siempre tiene fondo
- **Adecuado para:** Fotos, imágenes de tonos continuos

**Calidad de trazado:** (Buena para fotos, pero compleja)

**Mejores prácticas:**

- Use JPEG de alta calidad (baja compresión)
- Aumente el contraste antes de importar
- Considere el preprocesamiento en un editor de imágenes
- Mejor convertir a PNG primero si es posible

---

### BMP (Mapa de bits)

**Extensión:** `.bmp`
**Tipo MIME:** `image/bmp`
**Importación:** Traza a vectores
**Exportación:** No compatible

**Características:**

- **Sin compresión** - Tamaños de archivo grandes
- **Formato simple** - Ampliamente compatible
- **Adecuado para:** Gráficos simples, salida de software antiguo

**Calidad de trazado:** (Buena, pero no mejor que PNG)

**Mejores prácticas:**

- Convierta a PNG para un tamaño de archivo menor (sin diferencia de calidad)
- Use solo si el software de origen no puede exportar PNG/SVG

---

## Páginas relacionadas

- [Importar archivos](importing) - Cómo importar cada formato
- [Exportar](exporting) - Opciones de exportación de G-code
