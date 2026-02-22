# Formatos de Archivo Soportados

Esta página proporciona información detallada sobre todos los formatos de archivo soportados por Rayforge, incluyendo capacidades, limitaciones y recomendaciones.

## Resumen de Formatos

### Referencia Rápida

| Formato              | Tipo     | Importar   | Exportar          | Uso Recomendado          |
| -------------------- | -------- | ---------- | ----------------- | ------------------------ |
| **SVG**              | Vector   | ✓ Directo  | ✓ Exportar objeto | Formato de diseño principal |
| **DXF**              | Vector   | ✓ Directo  | ✓ Exportar objeto | Intercambio CAD          |
| **PDF**              | Mixto    | ✓ Trazar   | –                 | Exportación de documento (limitado) |
| **PNG**              | Raster   | ✓ Trazar   | –                 | Fotos, imágenes          |
| **JPEG**             | Raster   | ✓ Trazar   | –                 | Fotos                    |
| **BMP**              | Raster   | ✓ Trazar   | –                 | Gráficos simples         |
| **RFS**              | Bosquejo | ✓ Directo  | ✓ Exportar objeto | Bosquejos paramétricos   |
| **Código G**         | Control  | –          | ✓ Principal       | Salida de máquina        |
| **Proyecto Rayforge**| Proyecto | ✓          | ✓                 | Guardar/cargar proyectos |

---

## Formatos Vectoriales

### SVG (Gráficos Vectoriales Escalables)

**Extensión:** `.svg`
**Tipo MIME:** `image/svg+xml`
**Importar:** Análisis vectorial directo o trazar mapa de bits
**Exportar:** Exportar objeto (solo geometría)

**¿Qué es SVG?**

SVG es un formato de imagen vectorial basado en XML. Es el **formato preferido** para importar diseños a Rayforge.

**Funciones Soportadas:**

- ✓ Trayectorias (líneas, curvas, arcos)
- ✓ Formas básicas (rectángulos, círculos, elipses, polígonos)
- ✓ Grupos y transformaciones
- ✓ Colores de trazo y relleno
- ✓ Múltiples capas
- ✓ Transformaciones de coordenadas (trasladar, rotar, escalar)

**Funciones No Soportadas/Limitadas:**

- ✗ Texto (debe convertirse a trayectorias primero)
- ✗ Degradados (simplificados o ignorados)
- ✗ Filtros y efectos (ignorados)
- ✗ Máscaras y trayectorias de recorte (pueden no funcionar correctamente)
- ✗ Imágenes rasterizadas embebidas (importadas separadamente si es posible)
- ✗ Estilos de trazo complejos (guiones pueden simplificarse)
- ✗ Símbolos y elementos use (instancias pueden no actualizarse)

**Notas de Exportación:**

Al exportar una pieza a SVG, Rayforge exporta la geometría como trayectorias vectoriales con:

- Renderizado solo de trazo (sin relleno)
- Unidades de milímetros
- Color de trazo negro

**Mejores Prácticas:**

1. **Usar formato SVG Simple** (no SVG Inkscape u otras variantes específicas de herramienta)
2. **Convertir texto a trayectorias** antes de exportar
3. **Simplificar trayectorias complejas** para reducir el conteo de nodos
4. **Aplanar grupos** cuando sea posible
5. **Eliminar elementos no usados** (guías, cuadrículas, capas ocultas)
6. **Establecer unidades del documento** a mm (unidad nativa de Rayforge)

**Recomendaciones de Software:**

- **Inkscape** (gratuito) - Excelente soporte SVG, formato nativo

---

### DXF (Formato de Intercambio de Dibujos)

**Extensión:** `.dxf`
**Tipo MIME:** `application/dxf`, `image/vnd.dxf`
**Importar:** Análisis vectorial directo
**Exportar:** Exportar objeto (solo geometría)

**¿Qué es DXF?**

DXF es un formato de dibujo de AutoCAD, ampliamente usado para intercambio CAD.

**Versiones Soportadas:**

- ✓ **R12/LT2** (recomendado - mejor compatibilidad)
- ✓ R13, R14
- ✓ R2000 y posteriores (generalmente funciona, pero R12 es más seguro)

**Entidades Soportadas:**

- ✓ Líneas (LINE)
- ✓ Polilíneas (LWPOLYLINE, POLYLINE)
- ✓ Arcos (ARC)
- ✓ Círculos (CIRCLE)
- ✓ Splines (SPLINE) - convertidas a polilíneas
- ✓ Elipses (ELLIPSE)
- ✓ Capas

**Funciones No Soportadas/Limitadas:**

- ✗ Entidades 3D (usar proyección 2D)
- ✗ Cotas y anotaciones (ignoradas)
- ✗ Bloques/inserciones (pueden no instanciar correctamente)
- ✗ Tipos de línea complejos (simplificados a sólido)
- ✗ Texto (ignorado, convertir a contornos primero)
- ✗ Sombras (pueden simplificarse o ignorarse)

**Notas de Exportación:**

Al exportar una pieza a DXF, Rayforge exporta:

- Líneas como entidades LWPOLYLINE
- Arcos como entidades ARC
- Curvas Bezier como entidades SPLINE
- Unidades de milímetros (INSUNITS = 4)

---

### RFS (Bosquejo Rayforge)

**Extensión:** `.rfs`
**Tipo MIME:** `application/x-rayforge-sketch`
**Importar:** Directo (piezas basadas en bosquejo)
**Exportar:** Exportar objeto (piezas basadas en bosquejo)

**¿Qué es RFS?**

RFS es el formato de bosquejo paramétrico nativo de Rayforge. Preserva todos los elementos
geométricos y restricciones paramétricas, permitiéndote guardar y compartir completamente
bosquejos editables.

**Funciones:**

- ✓ Todos los elementos geométricos (líneas, arcos, círculos, rectángulos, etc.)
- ✓ Todas las restricciones paramétricas
- ✓ Valores dimensionales y expresiones
- ✓ Áreas de relleno

**Cuándo Usar:**

- Guardar diseños paramétricos reutilizables
- Compartir bosquejos editables con otros usuarios de Rayforge
- Archivar trabajo en progreso

---

### PDF (Formato de Documento Portátil)

**Extensión:** `.pdf`
**Tipo MIME:** `application/pdf`
**Importar:** Renderizado a mapa de bits, luego trazado
**Exportar:** No soportado

**¿Qué es la Importación PDF?**

Rayforge puede importar archivos PDF rasterizándolos primero, luego trazando a vectores.

**Proceso:**

1. PDF renderizado a imagen rasterizada (300 DPI por defecto)
2. Raster trazado para crear trayectorias vectoriales
3. Trayectorias añadidas al documento

**Limitaciones:**

- **No es importación vectorial verdadera** - Incluso PDFs vectoriales se rasterizan
- **Pérdida de calidad** por rasterización
- **Solo primera página** - PDFs de múltiples páginas solo importan la página 1
- **Lento para PDFs complejos** - Renderizar y trazar toma tiempo

**Cuándo Usar:**

- Último recurso cuando SVG/DXF no están disponibles
- Importación rápida de diseños simples
- Documentos con contenido mixto

**Mejores Alternativas:**

- **Exportar SVG desde la fuente** en lugar de PDF
- **Usar formatos vectoriales** (SVG, DXF) cuando sea posible
- **Para texto:** Exportar con texto convertido a contornos

---

## Formatos Rasterizados

Todos los formatos rasterizados se **importan trazando** - convertidos automáticamente a trayectorias vectoriales.

### PNG (Gráficos de Red Portátiles)

**Extensión:** `.png`
**Tipo MIME:** `image/png`
**Importar:** Trazar a vectores
**Exportar:** No soportado

**Características:**

- **Compresión sin pérdida** - Sin pérdida de calidad
- **Soporte de transparencia** - Canal alfa preservado
- **Bueno para:** Logos, arte lineal, capturas de pantalla, cualquier cosa que necesite transparencia

**Calidad de Trazado:** (Excelente para imágenes de alto contraste)

**Mejores Prácticas:**

- Usa PNG para logos y gráficos con bordes nítidos
- Asegura alto contraste entre primer plano y fondo
- El fondo transparente funciona mejor que blanco

---

### JPEG (Grupo de Expertos Fotográficos Conjunto)

**Extensión:** `.jpg`, `.jpeg`
**Tipo MIME:** `image/jpeg`
**Importar:** Trazar a vectores
**Exportar:** No soportado

**Características:**

- **Compresión con pérdida** - Alguna pérdida de calidad
- **Sin transparencia** - Siempre tiene fondo
- **Bueno para:** Fotos, imágenes de tono continuo

**Calidad de Trazado:** (Bueno para fotos, pero complejo)

**Mejores Prácticas:**

- Usa JPEG de alta calidad (baja compresión)
- Aumenta el contraste antes de importar
- Considera pre-procesar en editor de imágenes
- Mejor convertir a PNG primero si es posible

---

### BMP (Mapa de Bits)

**Extensión:** `.bmp`
**Tipo MIME:** `image/bmp`
**Importar:** Trazar a vectores
**Exportar:** No soportado

**Características:**

- **Sin comprimir** - Tamaños de archivo grandes
- **Formato simple** - Ampliamente compatible
- **Bueno para:** Gráficos simples, salida de software antiguo

**Calidad de Trazado:** (Bueno, pero no mejor que PNG)

**Mejores Prácticas:**

- Convierte a PNG para tamaño de archivo más pequeño (sin diferencia de calidad)
- Solo usa si el software fuente no puede exportar PNG/SVG

---

## Páginas Relacionadas

- [Importando Archivos](importing) - Cómo importar cada formato
- [Exportando](exporting) - Opciones de exportación de código G
