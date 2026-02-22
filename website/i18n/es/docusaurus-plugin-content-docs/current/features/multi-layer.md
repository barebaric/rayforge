# Flujo de Trabajo Multi-Capa

El sistema multi-capa de Rayforge te permite organizar trabajos complejos en etapas de procesamiento separadas, cada una con sus propias operaciones y ajustes. Esto es esencial para combinar diferentes procesos como grabado y corte, o trabajar con múltiples materiales.

## ¿Qué Son las Capas?

Una **capa** en Rayforge es:

- **Un contenedor** para piezas de trabajo (formas importadas, imágenes, texto)
- **Un flujo de trabajo** que define cómo se procesan esas piezas de trabajo
- **Un paso** procesado secuencialmente durante los trabajos

**Concepto clave:** Las capas se procesan en orden, una tras otra, permitiéndote controlar la secuencia de operaciones.

:::note Capas y Piezas de Trabajo
Una capa contiene una o más piezas de trabajo. Al importar archivos SVG con capas, cada capa de tu diseño se convierte en una capa separada en Rayforge. Esto te permite mantener tu diseño organizado exactamente como lo creaste.
:::


---

## ¿Por Qué Usar Múltiples Capas?

### Casos de Uso Comunes

**1. Grabar y Luego Cortar**

El flujo de trabajo multi-capa más común:

- **Capa 1:** Grabar raster el diseño
- **Capa 2:** Cortar contorno del borde

**¿Por qué capas separadas?**

- Grabar primero asegura que la pieza no se mueva durante el grabado
- Cortar al final previene que las piezas caigan antes de completar el grabado
- Diferentes ajustes de potencia/velocidad para cada operación

**2. Corte Multi-Pasada**

Para materiales gruesos:

- **Capa 1:** Primera pasada a potencia moderada
- **Capa 2:** Segunda pasada a potencia completa (misma geometría)
- **Capa 3:** Tercera pasada opcional si es necesario

**Beneficios:**

- Reduce chamuscado comparado con una sola pasada de alta potencia
- Cada capa puede tener diferentes ajustes de velocidad/potencia

**3. Proyectos Multi-Material**

Diferentes materiales en un trabajo:

- **Capa 1:** Cortar piezas de acrílico
- **Capa 2:** Grabar piezas de madera
- **Capa 3:** Marcar piezas de metal

**Requisitos:**

- Cada capa apunta a diferentes áreas de la cama
- Diferentes velocidad/potencia/enfoque para cada material

**4. Importación de Capas SVG**

Importar archivos SVG con estructura de capas existente:

- **Capa 1:** Elementos de grabado del SVG
- **Capa 2:** Elementos de corte del SVG
- **Capa 3:** Elementos de marcado del SVG

**Flujo de trabajo:**

- Importar un archivo SVG que tiene capas
- Habilitar "Usar Vectores Originales" en el diálogo de importación
- Seleccionar qué capas importar de la lista de capas detectadas
- Cada capa se convierte en una capa separada en Rayforge

**Requisitos:**

- Tu archivo SVG debe usar capas (creado en Inkscape o software similar)
- Habilitar "Usar Vectores Originales" al importar
- Los nombres de las capas se preservan de tu software de diseño

---

## Crear y Gestionar Capas

### Añadir una Nueva Capa

1. **Haz clic en el botón "+"** en el panel de Capas
2. **Nombra la capa** descriptivamente (ej., "Capa de Grabado", "Capa de Corte")
3. **La capa aparece** en la lista de capas

**Por defecto:** Los documentos nuevos comienzan con una capa.

### Propiedades de Capa

Cada capa tiene:

| Propiedad        | Descripción                                          |
| ---------------- | ---------------------------------------------------- |
| **Nombre**       | El nombre mostrado en la lista de capas              |
| **Visible**      | Alternar visibilidad en lienzo y previsualización    |
| **Material Base**| Asociación opcional de material                      |
| **Flujo de trabajo** | La(s) operación(es) aplicadas a piezas de trabajo en esta capa |
| **Piezas de trabajo** | Las formas/imagenes contenidas en esta capa      |

:::note Capas como Contenedores
Las capas son contenedores para tus piezas de trabajo. Al importar archivos SVG con capas, cada capa de tu diseño se convierte en una capa separada en Rayforge.
:::


### Reordenar Capas

**Orden de ejecución = orden de capas en la lista (de arriba a abajo)**

Para reordenar:

1. **Arrastra y suelta** capas en el panel de Capas
2. **El orden importa** - las capas se ejecutan de arriba a abajo

**Ejemplo:**

```
Panel de Capas:
1. Capa de Grabado     Se ejecuta primero
2. Capa de Marcado     Se ejecuta segundo
3. Capa de Corte       Se ejecuta último (recomendado)
```

### Eliminar Capas

1. **Selecciona la capa** en el panel de Capas
2. **Haz clic en el botón eliminar** o presiona Eliminar
3. **Confirma la eliminación** (todas las piezas de trabajo en la capa son removidas)

:::warning La Eliminación es Permanente
Eliminar una capa remueve todas sus piezas de trabajo y ajustes de flujo de trabajo. Usa Deshacer si eliminas accidentalmente.
:::


---

## Asignar Piezas de Trabajo a Capas

### Asignación Manual

1. **Importa o crea** una pieza de trabajo
2. **Arrastra la pieza de trabajo** a la capa deseada en el panel de Capas
3. **O usa el panel de propiedades** para cambiar la capa de la pieza de trabajo

### Importación de Capas SVG

Al importar archivos SVG con "Usar Vectores Originales" habilitado:

1. **Habilita "Usar Vectores Originales"** en el diálogo de importación
2. **Rayforge detecta capas** de tu archivo SVG
3. **Selecciona qué capas** importar usando los interruptores de capa
4. **Cada capa seleccionada** se convierte en una capa separada con su propia pieza de trabajo

:::note Detección de Capas
Rayforge detecta automáticamente capas de tu archivo SVG. Cada capa que creaste en tu software de diseño aparecerá como una capa separada en Rayforge.
:::


:::note Solo Importación Vectorial
La selección de capas solo está disponible al usar importación vectorial directa. Al usar modo trazo, el SVG completo se procesa como una pieza de trabajo.
:::


### Mover Piezas de Trabajo Entre Capas

**Arrastrar y soltar:**

- Selecciona pieza(s) de trabajo en el lienzo o panel de Documento
- Arrastra a la capa objetivo en el panel de Capas

**Cortar y pegar:**

- Corta pieza de trabajo de la capa actual (Ctrl+X)
- Selecciona la capa objetivo
- Pega (Ctrl+V)

### Diálogo de Importación SVG

Al importar archivos SVG, el diálogo de importación proporciona opciones que afectan el manejo de capas:

**Modo de Importación:**

- **Usar Vectores Originales:** Preserva tus trayectorias vectoriales y estructura de capas. Cuando está habilitado, aparece una sección "Capas" mostrando todas las capas de tu archivo.
- **Modo Trazo:** Convierte el SVG a un bitmap y traza los contornos. La selección de capas está deshabilitada en este modo.

**Sección de Capas (Solo Importación Vectorial):**

- Muestra todas las capas de tu archivo SVG
- Cada capa tiene un interruptor para habilitar/deshabilitar importación
- Los nombres de capas de tu software de diseño se preservan
- Solo las capas seleccionadas se importan como capas separadas

:::tip Preparar Archivos SVG para Importación de Capas
Para usar importación de capas SVG, crea tu diseño con capas en software como Inkscape. Usa el panel de Capas para organizar tu diseño, y Rayforge preservará esa estructura.
:::


---

## Flujos de Trabajo de Capa

Cada capa tiene un **Flujo de Trabajo** que define cómo se procesan sus piezas de trabajo.

### Configurar Flujos de Trabajo de Capa

Para cada capa, eliges un tipo de operación y configuras sus ajustes:

**Tipos de Operación:**

- **Contorno** - Sigue contornos (para corte o marcado)
- **Grabado Raster** - Graba imágenes y rellena áreas
- **Grabado de Profundidad** - Crea grabados de profundidad variable

**Mejoras Opcionales:**

- **Pestañas** - Pequeños puentes para mantener piezas en su lugar durante el corte
- **Overscan** - Extiende cortes más allá de la forma para bordes más limpios
- **Ajuste de Kerf** - Compensa el ancho de corte del láser

### Configuraciones Comunes de Capa

**Capa de Grabado:**

- Operación: Grabado Raster
- Ajustes: 300-500 DPI, velocidad moderada
- Típicamente no se necesitan opciones adicionales

**Capa de Corte:**

- Operación: Corte de Contorno
- Opciones: Pestañas (para mantener piezas), Overscan (para bordes limpios)
- Ajustes: Velocidad más lenta, mayor potencia

**Capa de Marcado:**

- Operación: Contorno (potencia ligera, no corta a través)
- Ajustes: Baja potencia, velocidad rápida
- Propósito: Líneas de doblez, líneas decorativas

---

## Visibilidad de Capas

Controla qué capas se muestran en el lienzo y previsualizaciones:

### Visibilidad en Lienzo

- **Icono de ojo** en panel de Capas alterna visibilidad
- **Capas ocultas:**
  - No se muestran en lienzo 2D
  - No se muestran en previsualización 3D
  - **Aún se incluyen en G-code generado**

**Casos de uso:**

- Ocultar capas de grabado complejas mientras posicionas capas de corte
- Despejar el lienzo cuando trabajas en capas específicas
- Enfocarte en una capa a la vez

### Visibilidad vs. Habilitado

| Estado                    | Lienzo | Previsualización | G-code |
| ------------------------- | ------ | ---------------- | ------ |
| **Visible y Habilitado**  | Sí     | Sí               | Sí     |
| **Oculto y Habilitado**   | No     | No               | Sí     |
| **Visible y Deshabilitado**| Sí    | Sí               | No     |
| **Oculto y Deshabilitado**| No     | No               | No     |

:::note Deshabilitar Capas
:::

Para excluir temporalmente una capa de trabajos sin eliminarla, desactiva la operación de la capa o deshabilítala en los ajustes de capa.

---

## Orden de Ejecución de Capas

### Cómo se Procesan las Capas

Durante la ejecución del trabajo, Rayforge procesa cada capa en orden de arriba a abajo. Dentro de cada capa, todas las piezas de trabajo se procesan antes de pasar a la siguiente capa.

### El Orden Importa

**Orden incorrecto:**

```
1. Capa de Corte
2. Capa de Grabado
```

**Problema:** ¡Las piezas cortadas pueden caerse o moverse antes del grabado!

**Orden correcto:**

```
1. Capa de Grabado
2. Capa de Corte
```

**Por qué:** El grabado ocurre mientras la pieza aún está adjunta, luego el corte la libera.

### Múltiples Pasadas

Para materiales gruesos, crea múltiples capas de corte:

```
1. Capa de Grabado
2. Capa de Corte (Pasada 1) - 50% potencia
3. Capa de Corte (Pasada 2) - 75% potencia
4. Capa de Corte (Pasada 3) - 100% potencia
```

**Consejo:** Usa la misma geometría para todas las pasadas de corte (duplica la capa).

---

## Técnicas Avanzadas

### Agrupación de Capas por Material

Usa capas para organizar por material al ejecutar trabajos mixtos:

```
Material 1 (Acrílico 3mm):
  - Capa de Grabado Acrílico
  - Capa de Corte Acrílico

Material 2 (Contrachapado 3mm):
  - Capa de Grabado Madera
  - Capa de Corte Madera
```

**Flujo de trabajo:**

1. Procesa todas las capas del Material 1
2. Cambia materiales
3. Procesa todas las capas del Material 2

**Alternativa:** Usa documentos separados para diferentes materiales.

### Pausar Entre Capas

Puedes configurar Rayforge para pausar entre capas. Esto es útil cuando necesitas:

- Cambiar materiales a mitad del trabajo
- Inspeccionar progreso antes de continuar
- Ajustar enfoque para diferentes operaciones

Para configurar pausas de capa, usa la función de hooks en los ajustes de tu máquina.

### Ajustes Específicos de Capa

El flujo de trabajo de cada capa puede tener ajustes únicos:

| Capa    | Operación | Velocidad  | Potencia | Pasadas |
| ------- | --------- | ---------- | -------- | ------- |
| Grabar  | Raster    | 300 mm/min | 20%      | 1       |
| Marcar  | Contorno  | 500 mm/min | 10%      | 1       |
| Cortar  | Contorno  | 100 mm/min | 90%      | 2       |

---

## Mejores Prácticas

### Convenciones de Nomenclatura

**Buenos nombres de capa:**

- "Grabado - Logo"
- "Corte - Contorno Exterior"
- "Marcado - Líneas de Doblez"
- "Pasada 1 - Corte Bruto"
- "Pasada 2 - Corte Final"

**Nombres de capa pobres:**

- "Capa 1", "Capa 2" (no descriptivo)
- Descripciones largas (mantén conciso)

### Organización de Capas

1. **De arriba a abajo = orden de ejecución**
2. **Grabado antes de corte** (regla general)
3. **Agrupa operaciones relacionadas** (todo corte, todo grabado)
4. **Usa visibilidad** para enfocarte en el trabajo actual
5. **Elimina capas no usadas** para mantener proyectos limpios

### Preparar Archivos SVG para Importación de Capas

**Para mejores resultados al importar capas SVG:**

1. **Usa el panel de Capas** en tu software de diseño para organizar tu diseño
2. **Asigna nombres significativos** a cada capa (ej., "Grabado", "Corte")
3. **Mantén capas planas** - evita poner capas dentro de otras capas
4. **Guarda tu archivo** e importa en Rayforge
5. **Verifica detección de capas** revisando el diálogo de importación

Rayforge funciona mejor con archivos SVG creados en Inkscape o software similar de diseño vectorial que soporte capas.

### Rendimiento

**Muchas capas:**

- Sin impacto significativo en rendimiento
- 10-20 capas es común para trabajos complejos
- Organiza lógicamente, no para minimizar conteo de capas

**Simplifica si es necesario:**

- Combina operaciones similares en una capa cuando sea posible
- Usa menos grabados raster (los más intensivos en recursos)

---

## Solución de Problemas

### La Capa No Genera G-code

**Problema:** La capa aparece en el documento pero no en el G-code generado.

**Soluciones:**

1. **Verifica que la capa tenga piezas de trabajo** - Las capas vacías se saltan
2. **Verifica que el flujo de trabajo esté configurado** - La capa necesita una operación
3. **Verifica ajustes de operación** - Potencia > 0, velocidad válida, etc.
4. **Verifica visibilidad de pieza de trabajo** - Piezas ocultas pueden no procesarse
5. **Regenera G-code** - Haz un pequeño cambio para forzar regeneración

### Orden de Capas Incorrecto

**Problema:** Las operaciones se ejecutan en orden inesperado.

**Solución:** Reordena capas en el panel de Capas. Recuerda: arriba = primero.

### Capas Superpuestas en Previsualización

**Problema:** Múltiples capas muestran contenido superpuesto en previsualización.

**Aclaración:** Esto es normal si las capas comparten la misma área XY.

**Soluciones:**

- Usa visibilidad de capa para ocultar otras capas temporalmente
- Revisa previsualización 3D para ver profundidad/orden
- Verifica que esto sea intencional (ej., grabar y luego cortar la misma forma)

### Pieza de Trabajo en Capa Incorrecta

**Problema:** La pieza de trabajo fue asignada a capa incorrecta.

**Solución:** Arrastra pieza de trabajo a capa correcta en panel de Capas o árbol de Documento.

### Capas SVG No Detectadas

**Problema:** Importando un archivo SVG pero no aparecen capas en el diálogo de importación.

**Soluciones:**

1. **Verifica estructura SVG** - Abre tu archivo en Inkscape o software similar para verificar que tiene capas
2. **Habilita "Usar Vectores Originales"** - La selección de capas solo está disponible en este modo de importación
3. **Verifica que tu diseño tiene capas** - Asegúrate de crear capas en tu software de diseño, no solo grupos
4. **Busca capas anidadas** - Capas dentro de otras capas pueden no detectarse correctamente
5. **Vuelve a guardar tu archivo** - A veces volver a guardar con una versión actual de tu software de diseño ayuda

### Importación de Capa SVG Muestra Contenido Incorrecto

**Problema:** La capa importada muestra contenido de otras capas o está vacía.

**Soluciones:**

1. **Verifica selección de capa** - Confirma que las capas correctas están habilitadas en el diálogo de importación
2. **Verifica tu diseño** - Abre el archivo original en tu software de diseño para confirmar que cada capa contiene el contenido correcto
3. **Busca elementos compartidos** - Elementos que aparecen en múltiples capas pueden causar confusión
4. **Prueba modo trazo** - Usa modo trazo como alternativa si la importación vectorial tiene problemas

---

## Páginas Relacionadas

- [Operaciones](./operations/contour) - Tipos de operación para flujos de trabajo de capa
- [Modo Simulación](./simulation-mode) - Previsualizar ejecución multi-capa
- [Macros y Hooks](../machine/hooks-macros) - Hooks a nivel de capa para automatización
- [Previsualización 3D](../ui/3d-preview) - Visualizar pila de capas
