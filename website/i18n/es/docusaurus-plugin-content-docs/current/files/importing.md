# Importar archivos

Rayforge admite la importación de varios formatos de archivo, tanto vectoriales
como de mapa de bits. Esta página explica cómo importar archivos y optimizarlos
para obtener los mejores resultados.

## Formatos de archivo admitidos

### Formatos vectoriales

| Formato   | Extensión | Método de importación     | Mejor para                            |
| --------- | --------- | ------------------------- | ------------------------------------- |
| **SVG**   | `.svg`    | Vectores directos o trazo | Gráficos vectoriales, logos, diseños  |
| **DXF**   | `.dxf`    | Vectores directos         | Planos CAD, diseños técnicos          |
| **PDF**   | `.pdf`    | Vectores directos o trazo | Documentos con contenido vectorial    |
| **Ruida** | `.rd`     | Vectores directos         | Archivos de trabajo controlador Ruida |

### Formatos de mapa de bits

| Formato  | Extensión       | Método de importación | Mejor para                             |
| -------- | --------------- | --------------------- | -------------------------------------- |
| **PNG**  | `.png`          | Trazo a vectores      | Fotos, imágenes con transparencia      |
| **JPEG** | `.jpg`, `.jpeg` | Trazo a vectores      | Fotos, imágenes de tono continuo       |
| **BMP**  | `.bmp`          | Trazo a vectores      | Gráficos simples, capturas de pantalla |

:::note Importación de mapa de bits
:::

Todas las imágenes de mapa de bits se **trazan** para crear trazados vectoriales
que pueden utilizarse en operaciones láser. La calidad depende de la
configuración del trazado.

---

## Importar archivos

### El diálogo de importación

Rayforge cuenta con un diálogo de importación unificado que proporciona vista
previa en tiempo real y opciones de configuración para todos los tipos de
archivo admitidos. El diálogo permite:

- **Vista previa de la importación** antes de añadirla al documento
- **Configurar los ajustes de trazado** para imágenes de mapa de bits
- **Elegir el método de importación** para archivos SVG (vectores directos o
  trazo)
- **Ajustar parámetros** como umbral, invertir y umbral automático

![Diálogo de importación](/screenshots/import-dialog.png)

### Método 1: Menú Archivo

1. **Importar archivo** (o Ctrl+I)
2. **Seleccionar el archivo** desde el selector de archivos
3. **Configurar los ajustes de importación** en el diálogo de importación
4. **Vista previa** del resultado antes de importar
5. **Pulsar Importar** para añadir al lienzo y al árbol del documento

### Método 2: Arrastrar y soltar

1. **Arrastrar el archivo** desde el gestor de archivos
2. **Soltar sobre** el lienzo de Rayforge
3. **Configurar los ajustes de importación** en el diálogo de importación
4. **Vista previa** del resultado antes de importar
5. **Pulsar Importar** para añadir al lienzo y al árbol del documento

### Método 3: Línea de comandos

```bash
# Abrir Rayforge con un archivo
rayforge myfile.svg

# Múltiples archivos
rayforge file1.svg file2.dxf
```

### Auto-Redimensionar al importar

Al importar archivos más grandes que el área de trabajo de la máquina,
Rayforge automáticamente:

1. **Reduce la escala** del contenido importado para que quepa dentro de los
   límites de la máquina
2. **Preserva la relación de aspecto** durante el escalado
3. **Centra** el contenido escalado en el espacio de trabajo
4. **Muestra una notificación** con la opción de deshacer el redimensionado

La notificación de redimensionado aparece como un mensaje emergente:

- ⚠️ "El elemento importado era más grande que el área de trabajo y se ha
  reducido de escala para ajustarse."
- Incluye un botón **"Restablecer"** para deshacer el auto-redimensionado
- El mensaje permanece visible hasta que se descarta o se ejecuta la acción de
  restablecimiento

Esto garantiza que sus diseños siempre se ajusten a las capacidades de su
máquina, dándole la flexibilidad de restaurar el tamaño original si es necesario.

---

## Importación SVG

SVG (Scalable Vector Graphics) es el **formato recomendado** para diseños
vectoriales.

### Opciones de importación en el diálogo

Al importar SVG, el diálogo de importación proporciona un interruptor para
elegir entre dos métodos:

#### 1. Usar vectores originales (Recomendado)

Esta opción está habilitada por defecto en el diálogo de importación.

**Cómo funciona:**

- Analiza el SVG y convierte los trazados directamente a geometría de Rayforge
- Preservación de alta fidelidad de curvas y formas
- Mantiene los datos vectoriales exactos

**Ventajas:**

- Mejor calidad y precisión
- Trazados editables
- Tamaño de archivo reducido

**Desventajas:**

- Algunas funciones avanzadas de SVG no son compatibles
- Los SVG complejos pueden tener problemas

**Usar para:**

- Diseños vectoriales limpios de Inkscape, Illustrator
- Complejidad de simple a moderada
- Diseños sin funciones avanzadas de SVG

#### 2. Trazar mapa de bits

Desactive "Usar vectores originales" para utilizar este método.

**Cómo funciona:**

- Renderiza el SVG primero como imagen de mapa de bits
- Traza la imagen renderizada para crear vectores
- Más compatible pero menos preciso

**Ventajas:**

- Maneja funciones SVG complejas
- Método de respaldo robusto
- Admite efectos y filtros

**Desventajas:**

- Pérdida de calidad por rasterización
- Tamaños de archivo mayores
- Menos preciso

**Usar para:**

- SVG que fallan en la importación directa
- SVG con efectos, filtros, degradados
- Cuando la importación directa produce errores

### Vista previa en tiempo real

El diálogo de importación muestra una vista previa en tiempo real de cómo se
importará su SVG:

- Los trazados vectoriales se muestran en una superposición azul
- En modo de trazo, se muestra la imagen original con los trazados trazados
- La vista previa se actualiza en tiempo real al cambiar los ajustes

### Buenas prácticas SVG

**Prepare su SVG para obtener los mejores resultados:**

1. **Convertir texto a trazados:**
   - Inkscape: `Trazado → Objeto a trazado`
   - Illustrator: `Texto → Crear contornos`

2. **Simplificar trazados complejos:**
   - Inkscape: `Trazado → Simplificar` (Ctrl+L)
   - Eliminar nodos innecesarios

3. **Desagrupar grupos anidados:**
   - Aplanar la jerarquía donde sea posible
   - `Objeto → Desagrupar` (Ctrl+Shift+G)

4. **Eliminar elementos ocultos:**
   - Borrar guías, cuadrículas, líneas de construcción
   - Eliminar objetos invisibles/transparentes

5. **Guardar como SVG simple:**
   - Inkscape: "SVG simple" o "SVG optimizado"
   - No "SVG de Inkscape" (contiene metadatos adicionales)

6. **Verificar las unidades del documento:**
   - Establecer en mm o pulgadas según corresponda
   - Rayforge utiliza mm internamente

**Funciones SVG comunes que pueden no importarse:**

- Degradados (convertir a rellenos sólidos o mapa de bits)
- Filtros y efectos (aplanar a trazados)
- Máscaras y trazados de recorte (expandir/aplanar)
- Imágenes de mapa de bits incrustadas (exportar por separado)
- Texto (convertir a trazados primero)

---

## Importación DXF

DXF (Drawing Exchange Format) es común en software CAD.

### Versiones DXF

Rayforge admite formatos DXF estándar:

- **R12/LT2** (recomendado) - Mejor compatibilidad
- **R13, R14** - Buen soporte
- **R2000+** - Generalmente funciona, pero R12 es más seguro

**Consejo:** Exporte como DXF R12/LT2 para máxima compatibilidad.

### Consejos para importación DXF

**Antes de exportar desde CAD:**

1. **Simplificar el dibujo:**
   - Eliminar capas innecesarias
   - Borrar cotas y anotaciones
   - Eliminar objetos 3D (usar proyección 2D)

2. **Verificar unidades:**
   - Comprobar las unidades del dibujo (mm vs pulgadas)
   - Rayforge asume mm por defecto

3. **Aplanar capas:**
   - Considerar exportar solo las capas relevantes
   - Ocultar o eliminar capas de construcción

4. **Usar precisión adecuada:**
   - La precisión láser es típicamente de 0,1 mm
   - No especifique demasiada precisión

**Después de importar:**

- Verificar la escala (las unidades DXF pueden necesitar ajuste)
- Comprobar que todos los trazados se importaron correctamente
- Eliminar elementos de construcción no deseados

---

## Importación PDF

Los archivos PDF pueden contener gráficos vectoriales, imágenes de mapa de
bits, o ambos.

### Importación directa de vectores

Al importar un PDF que contiene trazados vectoriales, Rayforge puede importarlos
directamente, igual que archivos SVG o DXF. Esto proporciona geometría limpia y
escalable sin pérdida de calidad por rasterización.

Si el PDF contiene capas, Rayforge las detecta y permite seleccionar cuáles
importar. Cada capa se convierte en una pieza de trabajo separada en su
documento. Esto funciona de la misma manera que la importación de capas SVG:
habilite o deshabilite capas individuales en el diálogo de importación antes de
importar.

Esto es especialmente útil para PDF exportados desde software de diseño como
Illustrator o Inkscape, donde los trazados vectoriales son limpios y están bien
organizados.

### Alternativa: Renderizar y trazar

Para PDF que no contienen datos vectoriales utilizables — documentos escaneados,
fotos incrustadas, o PDF donde el texto no se ha convertido a contornos —
Rayforge puede recurrir a renderizar el PDF como imagen y trazarlo. Esto
funciona igual que la importación de imágenes de mapa de bits.

### Consejos para importación PDF

**Mejores resultados:**

1. **Usar PDF vectoriales**: Los PDF creados desde software vectorial (Illustrator,
   Inkscape) producen los resultados más limpios con importación directa.

2. **Verificar capas**: Si su PDF tiene capas, las verá listadas en el diálogo
   de importación. Seleccione solo las capas que necesite.

3. **Para documentos con texto**: Exporte como SVG con fuentes convertidas a
   trazados para la mejor calidad, o use la alternativa de renderizar y trazar.

4. **Usar la vista previa del diálogo de importación**: Ajuste los ajustes de
   umbral e invertir al usar el modo de trazo. La vista previa muestra
   exactamente cómo se trazará el PDF.

---

## Importación Ruida

Los archivos Ruida (.rd) son archivos de trabajo binarios propietarios utilizados
por los controladores Ruida en muchas máquinas de corte láser. Estos archivos
contienen tanto geometría vectorial como ajustes del láser organizados en capas
(colores).

**Después de importar:**

- **Verificar escala** - Comprobar que las dimensiones coinciden con el tamaño
  esperado
- **Revisar capas** - Asegurar que todas las capas se importaron correctamente
- **Validar trazados** - Confirmar que todos los trazados de corte están
  presentes

### Limitaciones

- **Importación de solo lectura** - Los archivos Ruida solo pueden importarse,
  no exportarse
- **Formato binario** - La edición directa de archivos .rd originales no es
  compatible
- **Funciones propietarias** - Algunas funciones avanzadas de Ruida pueden no ser
  totalmente compatibles

---

## Importación de imágenes de mapa de bits (PNG, JPG, BMP)

Las imágenes de mapa de bits se **trazan** para crear trazados vectoriales
utilizando el diálogo de importación.

### Proceso de trazado en el diálogo

**Cómo funciona:**

1. **Imagen cargada** en el diálogo de importación
2. **Vista previa en tiempo real** muestra el resultado del trazado
3. **Ajustes de trazado** pueden modificarse en tiempo real
4. **Trazados vectoriales creados** a partir de los bordes trazados
5. **Trazados añadidos** al documento como piezas de trabajo al importar

### Configuración de trazado en el diálogo

El diálogo de importación proporciona estos parámetros ajustables:

| Parámetro             | Descripción          | Efecto                                                     |
| --------------------- | -------------------- | ---------------------------------------------------------- |
| **Umbral automático** | Detección automática | Al habilitarse, encuentra automáticamente el umbral óptimo |
| **Umbral**            | Corte blanco/negro   | Menor = más detalle, mayor = más simple                    |
| **Invertir**          | Invertir colores     | Traza objetos claros sobre fondo oscuro                    |

**Los ajustes predeterminados** funcionan bien para la mayoría de las imágenes.
El diálogo muestra una vista previa en tiempo real que se actualiza al ajustar
estos parámetros, permitiéndole afinar el trazado antes de importar.

### Preparar imágenes para trazado

**Para obtener los mejores resultados:**

1. **Alto contraste:**
   - Ajustar brillo/contraste en un editor de imágenes
   - Distinción clara entre primer plano y fondo

2. **Fondo limpio:**
   - Eliminar ruido y artefactos
   - Fondo blanco sólido o transparente

3. **Resolución adecuada:**
   - 300-500 PPP para fotos
   - Demasiado alta = trazado lento, demasiado baja = mala calidad

4. **Recortar al contenido:**
   - Eliminar bordes innecesarios
   - Enfocarse en el área a grabar/cortar

5. **Convertir a blanco y negro:**
   - Para corte: blanco y negro puro
   - Para grabado: escala de grises es aceptable

**Herramientas de edición de imágenes:**

- GIMP (gratuito)
- Photoshop
- Krita (gratuito)
- Paint.NET (gratuito, Windows)

### Calidad del trazado

**Buenos candidatos para trazo:**

- Logos con bordes definidos
- Imágenes de alto contraste
- Arte lineal y dibujos
- Texto (aunque es mejor como vector)

**Malos candidatos para trazo:**

- Imágenes de baja resolución
- Fotos con bordes suaves
- Imágenes con degradados
- Fotos muy detalladas o complejas

---

## Páginas relacionadas

- [Formatos admitidos](formats) - Especificaciones detalladas de formatos
- [Exportar G-code](exporting) - Opciones de salida
- [Inicio rápido](../getting-started/quick-start) - Tutorial de primera importación
