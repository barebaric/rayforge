# Importando Archivos

Rayforge soporta importar varios formatos de archivo, tanto vectoriales como rasterizados. Esta página explica cómo importar archivos y optimizarlos para obtener mejores resultados.

## Formatos de Archivo Soportados

### Formatos Vectoriales

| Formato   | Extensión | Método de Importación       | Mejor Para                        |
| --------- | --------- | --------------------------- | --------------------------------- |
| **SVG**   | `.svg`    | Vectores directos o trazar  | Gráficos vectoriales, logos, diseños |
| **DXF**   | `.dxf`    | Vectores directos           | Dibujos CAD, diseños técnicos     |
| **PDF**   | `.pdf`    | Renderizar y trazar         | Documentos con contenido vectorial |
| **Ruida** | `.rd`     | Vectores directos           | Archivos de trabajo de controlador Ruida |

### Formatos Rasterizados

| Formato   | Extensión       | Método de Importación | Mejor Para                         |
| --------- | --------------- | --------------------- | ---------------------------------- |
| **PNG**   | `.png`          | Trazar a vectores     | Fotos, imágenes con transparencia  |
| **JPEG**  | `.jpg`, `.jpeg` | Trazar a vectores     | Fotos, imágenes de tono continuo   |
| **BMP**   | `.bmp`          | Trazar a vectores     | Gráficos simples, capturas de pantalla |

:::note Importación de Rasterizados
:::

Todas las imágenes rasterizadas se **trazan** para crear trayectorias vectoriales que pueden usarse para operaciones láser. La calidad depende de la configuración de trazado.

---

## Importando Archivos

### El Diálogo de Importación

Rayforge presenta un diálogo de importación unificado que proporciona vista previa en vivo y
opciones de configuración para todos los tipos de archivo soportados. El diálogo te permite:

- **Previsualizar tu importación** antes de añadirla al documento
- **Configurar ajustes de trazado** para imágenes rasterizadas
- **Elegir el método de importación** para archivos SVG (vectores directos o trazar)
- **Ajustar parámetros** como umbral, invertir, y auto-umbral

![Diálogo de Importación](/screenshots/import-dialog.png)

### Método 1: Menú Archivo

1. **Archivo Importar** (o Ctrl+I)
2. **Selecciona tu archivo** del selector de archivos
3. **Configura los ajustes de importación** en el diálogo de importación
4. **Previsualiza** el resultado antes de importar
5. **Haz clic en Importar** para añadir al lienzo y árbol del documento

### Método 2: Arrastrar y Soltar

1. **Arrastra el archivo** desde tu gestor de archivos
2. **Suelta en** el lienzo de Rayforge
3. **Configura los ajustes de importación** en el diálogo de importación
4. **Previsualiza** el resultado antes de importar
5. **Haz clic en Importar** para añadir al lienzo y árbol del documento

### Método 3: Línea de Comandos

```bash
# Abrir Rayforge con un archivo
rayforge myfile.svg

# Múltiples archivos
rayforge file1.svg file2.dxf
```

### Auto-Redimensionar al Importar

Al importar archivos que son más grandes que el área de trabajo de tu máquina, Rayforge
automáticamente:

1. **Escala hacia abajo** el contenido importado para que quepa dentro de los límites de la máquina
2. **Preserva la relación de aspecto** durante el escalado
3. **Centra** el contenido escalado en el espacio de trabajo
4. **Muestra una notificación** con la opción de deshacer el redimensionamiento

La notificación de redimensionamiento aparece como un mensaje emergente:

- ⚠️ "El elemento importado era más grande que el área de trabajo y ha sido escalado para ajustarse."
- Incluye un botón **"Restablecer"** para deshacer el auto-redimensionamiento
- El mensaje permanece visible hasta que se descarta o se toma la acción de restablecimiento

Esto asegura que tus diseños siempre quepan dentro de las capacidades de tu máquina mientras te da
la flexibilidad de restaurar el tamaño original si es necesario.

---

## Importación SVG

SVG (Gráficos Vectoriales Escalables) es el **formato recomendado** para diseños vectoriales.

### Opciones de Importación en el Diálogo

Al importar SVG, el diálogo de importación proporciona un interruptor para elegir
entre dos métodos:

#### 1. Usar Vectores Originales (Recomendado)

Esta opción está habilitada por defecto en el diálogo de importación.

**Cómo funciona:**

- Analiza SVG y convierte trayectorias directamente a geometría de Rayforge
- Preservación de alta fidelidad de curvas y formas
- Mantiene datos vectoriales exactos

**Ventajas:**

- Mejor calidad y precisión
- Trayectorias editables
- Tamaño de archivo pequeño

**Desventajas:**

- Algunas funciones SVG avanzadas no soportadas
- SVGs complejos pueden tener problemas

**Usar para:**

- Diseños vectoriales limpios de Inkscape, Illustrator
- Complejidad simple a moderada
- Diseños sin funciones SVG avanzadas

#### 2. Trazar Mapa de Bits

Deshabilita "Usar Vectores Originales" para usar este método.

**Cómo funciona:**

- Renderiza SVG a una imagen rasterizada primero
- Traza la imagen renderizada para crear vectores
- Más compatible pero menos preciso

**Ventajas:**

- Maneja funciones SVG complejas
- Método de respaldo robusto
- Soporta efectos y filtros

**Desventajas:**

- Pérdida de calidad por rasterización
- Tamaños de archivo más grandes
- No tan preciso

**Usar para:**

- SVGs que fallan en la importación directa
- SVGs con efectos, filtros, degradados
- Cuando la importación directa produce errores

### Vista Previa en Vivo

El diálogo de importación muestra una vista previa en vivo de cómo se importará tu SVG:

- Las trayectorias vectoriales se muestran en superposición azul
- Para el modo de trazo, la imagen original se muestra con las trayectorias trazadas
- La vista previa se actualiza en tiempo real a medida que cambias los ajustes

### Mejores Prácticas SVG

**Prepara tu SVG para mejores resultados:**

1. **Convertir texto a trayectorias:**

   - Inkscape: `Trayectoria → Objeto a Trayectoria`
   - Illustrator: `Tipo → Crear Contornos`

2. **Simplificar trayectorias complejas:**

   - Inkscape: `Trayectoria → Simplificar` (Ctrl+L)
   - Eliminar nodos innecesarios

3. **Desagrupar grupos anidados:**

   - Aplanar jerarquía donde sea posible
   - `Objeto → Desagrupar` (Ctrl+Shift+G)

4. **Eliminar elementos ocultos:**

   - Eliminar guías, cuadrículas, líneas de construcción
   - Eliminar objetos invisibles/transparentes

5. **Guardar como SVG Simple:**

   - Inkscape: "SVG Simple" o "SVG Optimizado"
   - No "SVG Inkscape" (tiene metadatos extra)

6. **Verificar unidades del documento:**
   - Establecer a mm o pulgadas según sea apropiado
   - Rayforge usa mm internamente

**Funciones SVG comunes que pueden no importar:**

- Degradados (convertir a rellenos sólidos o raster)
- Filtros y efectos (aplanar a trayectorias)
- Máscaras y trayectorias de recorte (expandir/aplanar)
- Imágenes rasterizadas embebidas (exportar separadamente)
- Texto (convertir a trayectorias primero)

---

## Importación DXF

DXF (Formato de Intercambio de Dibujos) es común para software CAD.

### Versiones DXF

Rayforge soporta formatos DXF estándar:

- **R12/LT2** (recomendado) - Mejor compatibilidad
- **R13, R14** - Buen soporte
- **R2000+** - Generalmente funciona, pero R12 es más seguro

**Consejo:** Exporta como DXF R12/LT2 para máxima compatibilidad.

### Consejos de Importación DXF

**Antes de exportar desde CAD:**

1. **Simplificar el dibujo:**

   - Eliminar capas innecesarias
   - Eliminar cotas y anotaciones
   - Eliminar objetos 3D (usar proyección 2D)

2. **Verificar unidades:**

   - Verificar unidades del dibujo (mm vs pulgadas)
   - Rayforge asume mm por defecto

3. **Aplanar capas:**

   - Considera exportar solo capas relevantes
   - Ocultar o eliminar capas de construcción

4. **Usar precisión apropiada:**
   - La precisión láser es típicamente 0.1mm
   - No sobre-especifiques precisión

**Después de importar:**

- Verifica la escala (las unidades DXF pueden necesitar ajuste)
- Verifica que todas las trayectorias se importaron correctamente
- Elimina cualquier elemento de construcción no deseado

---

## Importación PDF

Los archivos PDF pueden contener gráficos vectoriales, imágenes rasterizadas, o ambos.

### Cómo Funciona la Importación PDF

Al importar archivos PDF a través del diálogo de importación, Rayforge **renderiza el PDF**
a una imagen, luego lo **traza** para crear vectores.

**Proceso:**

1. El PDF se renderiza y muestra en la vista previa del diálogo de importación
2. Puedes ajustar los ajustes de trazado en tiempo real
3. La imagen renderizada se traza usando vectorización con tus ajustes
4. Las trayectorias resultantes se añaden al documento cuando haces clic en Importar

**Limitaciones:**

- El texto se rasteriza (no editable como trayectorias)
- La calidad vectorial depende del DPI de renderizado
- PDFs de múltiples páginas: solo la primera página se importa

### Consejos de Importación PDF

**Mejores resultados:**

1. **Usar PDFs vectoriales:**

   - PDFs creados desde software vectorial (Illustrator, Inkscape)
   - No documentos escaneados o imágenes embebidas

2. **Exportar SVG en su lugar si es posible:**

   - La mayoría del software de diseño puede exportar SVG directamente
   - SVG tendrá mejor calidad que la importación PDF

3. **Para documentos con texto:**

   - Exportar como SVG con fuentes convertidas a trayectorias
   - O renderizar PDF a alto DPI (600+) y trazar

4. **Usar la vista previa del diálogo de importación:**
   - Ajusta los ajustes de umbral e invertir para mejores resultados
   - La vista previa muestra exactamente cómo se trazará el PDF

---

## Importación Ruida

Los archivos Ruida (.rd) son archivos de trabajo binarios propietarios usados por controladores Ruida en muchas máquinas de corte láser. Estos archivos contienen tanto geometría vectorial como ajustes de láser
organizados en capas (colores).

**Después de importar:**

- **Verifica la escala** - Confirma que las dimensiones coinciden con el tamaño esperado
- **Revisa las capas** - Asegúrate de que todas las capas se importaron correctamente
- **Valida trayectorias** - Confirma que todas las trayectorias de corte están presentes

### Limitaciones

- **Importación solo lectura** - Los archivos Ruida solo pueden importarse, no exportarse
- **Formato binario** - La edición directa de archivos .rd originales no soportada
- **Funciones propietarias** - Algunas funciones avanzadas de Ruida pueden no estar completamente soportadas

---

## Importación de Imágenes Rasterizadas (PNG, JPG, BMP)

Las imágenes rasterizadas se **trazan** para crear trayectorias vectoriales usando el diálogo de importación.

### Proceso de Trazado en el Diálogo

**Cómo funciona:**

1. **Imagen cargada** en el diálogo de importación
2. **Vista previa en vivo** muestra el resultado trazado
3. **Ajustes de trazado** pueden ajustarse en tiempo real
4. **Trayectorias vectoriales creadas** desde los bordes trazados
5. **Trayectorias añadidas** al documento como piezas cuando se importan

### Configuración de Trazado en el Diálogo

El diálogo de importación proporciona estos parámetros ajustables:

| Parámetro           | Descripción        | Efecto                                              |
| ------------------- | ------------------ | --------------------------------------------------- |
| **Auto Umbral**     | Detección automática | Cuando está habilitado, encuentra automáticamente el umbral óptimo |
| **Umbral**          | Corte negro/blanco | Menor = más detalle, mayor = más simple             |
| **Invertir**        | Invertir colores   | Trazar objetos claros sobre fondo oscuro            |

Los **ajustes predeterminados** funcionan bien para la mayoría de las imágenes. El diálogo muestra una vista previa en vivo
que se actualiza a medida que ajustas estos ajustes, permitiéndote afinar el trazo
antes de importar.

### Preparando Imágenes para Trazado

**Para mejores resultados:**

1. **Alto contraste:**

   - Ajusta brillo/contraste en editor de imágenes
   - Distinción clara entre primer plano y fondo

2. **Fondo limpio:**

   - Elimina ruido y artefactos
   - Fondo blanco sólido o transparente

3. **Resolución apropiada:**

   - 300-500 DPI para fotos
   - Demasiado alto = trazado lento, demasiado bajo = calidad pobre

4. **Recortar al contenido:**

   - Elimina bordes innecesarios
   - Enfócate en el área a grabar/cortar

5. **Convertir a blanco y negro:**
   - Para corte: B&W puro
   - Para grabado: escala de grises está bien

**Herramientas de edición de imágenes:**

- GIMP (gratuito)
- Photoshop
- Krita (gratuito)
- Paint.NET (gratuito, Windows)

### Calidad del Trazado

**Buenos candidatos para trazar:**

- Logos con bordes claros
- Imágenes de alto contraste
- Arte lineal y dibujos
- Texto (aunque mejor como vector)

**Malos candidatos para trazar:**

- Imágenes de baja resolución
- Fotos con bordes suaves
- Imágenes con degradados
- Fotos muy detalladas o complejas

---

## Páginas Relacionadas

- [Formatos Soportados](formats) - Especificaciones detalladas de formatos
- [Exportando Código G](exporting) - Opciones de salida
- [Inicio Rápido](../getting-started/quick-start) - Tutorial de primera importación
