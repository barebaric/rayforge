# Flujo de trabajo con múltiples capas

El sistema de múltiples capas de Rayforge le permite organizar trabajos complejos
en etapas de procesamiento separadas, cada una con sus propias operaciones y
ajustes. Esto es esencial para combinar diferentes procesos como grabado y corte,
o para trabajar con múltiples materiales.

## ¿Qué son las capas?

Una **capa** en Rayforge es:

- **Un contenedor** para piezas de trabajo (formas importadas, imágenes, texto)
- **Un flujo de trabajo** que define cómo se procesan esas piezas de trabajo
- **Un paso** procesado secuencialmente durante los trabajos

**Concepto clave:** Las capas se procesan en orden, una tras otra, lo que le
permite controlar la secuencia de operaciones.

:::note Capas y piezas de trabajo
Una capa contiene una o más piezas de trabajo. Al importar archivos SVG con
capas, cada capa de su diseño se convierte en una capa separada en Rayforge.
Esto le permite mantener su diseño organizado exactamente como lo creó.
:::

---

## ¿Por qué usar múltiples capas?

### Casos de uso comunes

**1. Grabar y luego cortar**

El flujo de trabajo con múltiples capas más común:

- **Capa 1:** Grabado ráster del diseño
- **Capa 2:** Corte de contorno del perfil

**¿Por qué capas separadas?**

- Grabar primero asegura que la pieza no se mueva durante el grabado
- Cortar al final evita que las piezas caigan antes de completar el grabado
- Diferentes ajustes de potencia/velocidad para cada operación

**2. Corte en múltiples pasadas**

Para materiales gruesos:

- **Capa 1:** Primera pasada a potencia moderada
- **Capa 2:** Segunda pasada a potencia máxima (misma geometría)
- **Capa 3:** Tercera pasada opcional si es necesario

**Beneficios:**

- Reduce el chamuscado en comparación con una sola pasada de alta potencia
- Cada capa puede tener diferentes ajustes de velocidad/potencia

**3. Proyectos con múltiples materiales**

Diferentes materiales en un trabajo:

- **Capa 1:** Cortar piezas de acrílico
- **Capa 2:** Grabar piezas de madera
- **Capa 3:** Marcar piezas de metal

**Requisitos:**

- Cada capa se dirige a diferentes áreas de la mesa de trabajo
- Diferentes velocidad/potencia/enfoque para cada material

**4. Importación de capas SVG**

Importar archivos SVG con estructura de capas existente:

- **Capa 1:** Elementos de grabado del SVG
- **Capa 2:** Elementos de corte del SVG
- **Capa 3:** Elementos de marcado del SVG

**Flujo de trabajo:**

- Importar un archivo SVG que tenga capas
- Activar "Usar vectores originales" en el diálogo de importación
- Seleccionar qué capas importar de la lista de capas detectadas
- Cada capa se convierte en una capa separada en Rayforge

**Requisitos:**

- Su archivo SVG debe usar capas (creadas en Inkscape o software similar)
- Activar "Usar vectores originales" al importar
- Los nombres de las capas se conservan desde su software de diseño

---

## Crear y gestionar capas

### Añadir una nueva capa

1. **Haga clic en el botón "+"** en el panel de Capas
2. **Asigne un nombre** descriptivo a la capa (ej.: "Capa de grabado",
   "Capa de corte")
3. **La capa aparece** en la lista de capas

**Predeterminado:** Los documentos nuevos comienzan con una capa.

### Propiedades de las capas

Cada capa tiene:

| Propiedad       | Descripción                                             |
| --------------- | ------------------------------------------------------- |
| **Nombre**      | El nombre mostrado en la lista de capas                 |
| **Visible**     | Alterna la visibilidad en el lienzo y la vista previa   |
| **Flujo de trabajo** | Las operaciones aplicadas a las piezas en esta capa |
| **Rotativo**    | Si esta capa se ejecuta en modo rotativo                |
| **Piezas de trabajo** | Las formas/imágenes contenidas en esta capa       |

### Modo rotativo por capa

Si tiene un [accesorio rotativo](../machine/rotary) configurado, puede activar
el modo rotativo para capas individuales. Esto le permite combinar trabajo en
superficie plana y cilíndrica en el mismo proyecto — por ejemplo, grabar un
diseño en la tapa plana de una caja en una capa y envolver texto alrededor del
cuerpo cilíndrico en otra.

Las capas con el modo rotativo activo muestran un pequeño icono rotativo en la
lista de capas. Cada capa recuerda su propio ajuste rotativo, por lo que puede
mezclarlas libremente.

:::note Capas como contenedores
Las capas son contenedores para sus piezas de trabajo. Al importar archivos SVG
con capas, cada capa de su diseño se convierte en una capa separada en Rayforge.
:::

### Reordenar capas

**El orden de ejecución = el orden de capas en la lista (de arriba a abajo)**

Para reordenar:

1. **Arrastre y suelte** las capas en el panel de Capas
2. **El orden importa** - las capas se ejecutan de arriba a abajo

**Ejemplo:**

```
Panel de Capas:
1. Capa de grabado     Se ejecuta primero
2. Capa de marcado     Se ejecuta segundo
3. Capa de corte       Se ejecuta al final (recomendado)
```

### Eliminar capas

1. **Seleccione la capa** en el panel de Capas
2. **Haga clic en el botón de eliminar** o pulse Suprimir
3. **Confirme la eliminación** (todas las piezas de trabajo de la capa se
   eliminan)

:::warning La eliminación es permanente
Eliminar una capa borra todas sus piezas de trabajo y ajustes de flujo de
trabajo. Use Deshacer si elimina accidentalmente.
:::

---

## Asignar piezas de trabajo a capas

### Asignación manual

1. **Importe o cree** una pieza de trabajo
2. **Arrastre la pieza de trabajo** a la capa deseada en el panel de Capas
3. **O use el panel de propiedades** para cambiar la capa de la pieza de
   trabajo

### Importación de capas SVG

Al importar archivos SVG con "Usar vectores originales" activado:

1. **Active "Usar vectores originales"** en el diálogo de importación
2. **Rayforge detecta las capas** de su archivo SVG
3. **Seleccione qué capas** importar usando los interruptores de capa
4. **Cada capa seleccionada** se convierte en una capa separada con su propia
   pieza

:::note Detección de capas
Rayforge detecta automáticamente las capas de su archivo SVG. Cada capa que
creó en su software de diseño aparecerá como una capa separada en Rayforge.
:::

:::note Solo importación de vectores
La selección de capas solo está disponible al usar importación directa de
vectores. Al usar el modo de trazado, el SVG completo se procesa como una sola
pieza de trabajo.
:::

### Mover piezas de trabajo entre capas

**Arrastrar y soltar:**

- Seleccione pieza(s) de trabajo en el lienzo o panel de Documento
- Arrastre a la capa de destino en el panel de Capas

**Cortar y pegar:**

- Corte la pieza de trabajo de la capa actual (Ctrl+X)
- Seleccione la capa de destino
- Pegue (Ctrl+V)

### Diálogo de importación SVG

Al importar archivos SVG, el diálogo de importación proporciona opciones que
afectan el manejo de capas:

**Modo de importación:**

- **Usar vectores originales:** Preserva sus trazados vectoriales y la
  estructura de capas. Cuando se activa, aparece una sección "Capas" que muestra
  todas las capas de su archivo.
- **Modo de trazado:** Convierte el SVG en un mapa de bits y traza los
  contornos. La selección de capas está desactivada en este modo.

**Sección de capadas (solo importación de vectores):**

- Muestra todas las capas de su archivo SVG
- Cada capa tiene un interruptor para activar/desactivar la importación
- Los nombres de las capas de su software de diseño se conservan
- Solo las capas seleccionadas se importan como capas separadas

:::tip Preparar archivos SVG para importación de capas
Para usar la importación de capas SVG, cree su diseño con capas en software como
Inkscape. Use el panel de Capas para organizar su diseño, y Rayforge conservará
esa estructura.
:::

---

## Flujos de trabajo de capas

Cada capa tiene un **flujo de trabajo** que define cómo se procesan sus piezas
de trabajo.

### Configurar flujos de trabajo de capas

Para cada capa, elige un tipo de operación y configura sus ajustes:

**Tipos de operación:**

- **Contorno** - Sigue los perfiles (para cortar o marcar)
- **Grabado ráster** - Graba imágenes y rellena áreas
- **Grabado con profundidad** - Crea grabados de profundidad variable

**Mejoras opcionales:**

- **Pestañas** - Pequeños puentes para mantener las piezas en su lugar durante
  el corte
- **Sobrescaneo** - Extiende los cortes más allá de la forma para bordes más
  limpios
- **Ajuste de kerf** - Compensa el ancho de corte del láser

### Configuraciones comunes de capas

**Capa de grabado:**

- Operación: Grabado ráster
- Ajustes: 300-500 DPI, velocidad moderada
- Normalmente no se necesitan opciones adicionales

**Capa de corte:**

- Operación: Corte de contorno
- Opciones: Pestañas (para sujetar piezas), Sobrescaneo (para bordes limpios)
- Ajustes: Velocidad más lenta, mayor potencia

**Capa de marcado:**

- Operación: Contorno (potencia ligera, no corta a través)
- Ajustes: Baja potencia, velocidad rápida
- Propósito: Líneas de pliegue, líneas decorativas

---

## Visibilidad de las capas

Controle qué capas se muestran en el lienzo y las vistas previas:

### Visibilidad en el lienzo

- **Icono de ojo** en el panel de Capas alterna la visibilidad
- **Capas ocultas:**
  - No se muestran en el lienzo 2D
  - No se muestran en la vista previa 3D
  - **Se incluyen en el G-code generado**

**Casos de uso:**

- Ocultar capas de grabado complejas mientras posiciona capas de corte
- Despejar el lienzo al trabajar en capas específicas
- Enfocarse en una capa a la vez

### Visibilidad vs. Activado

| Estado                    | Lienzo | Vista previa | G-code |
| ------------------------- | ------ | ------------ | ------ |
| **Visible y activado**    | Sí     | Sí           | Sí     |
| **Oculto y activado**     | No     | No           | Sí     |
| **Visible y desactivado** | Sí     | Sí           | No     |
| **Oculto y desactivado**  | No     | No           | No     |

:::note Desactivar capas
:::

Para excluir temporalmente una capa de los trabajos sin eliminarla, desactive la
operación de la capa o desactívela en los ajustes de la capa.

---

## Orden de ejecución de las capas

### Cómo se procesan las capas

Durante la ejecución del trabajo, Rayforge procesa cada capa en orden de arriba
a abajo. Dentro de cada capa, todas las piezas de trabajo se procesan antes de
pasar a la siguiente capa.

### El orden importa

**Orden incorrecto:**

```
1. Capa de corte
2. Capa de grabado
```

**Problema:** ¡Las piezas cortadas pueden caerse o moverse antes del grabado!

**Orden correcto:**

```
1. Capa de grabado
2. Capa de corte
```

**Por qué:** El grabado ocurre mientras la pieza aún está sujeta, luego el
corte la libera.

### Múltiples pasadas

Para materiales gruesos, cree múltiples capas de corte:

```
1. Capa de grabado
2. Capa de corte (Pasada 1) - 50% potencia
3. Capa de corte (Pasada 2) - 75% potencia
4. Capa de corte (Pasada 3) - 100% potencia
```

**Consejo:** Use la misma geometría para todas las pasadas de corte (duplicar la
capa).

---

## Técnicas avanzadas

### Agrupación de capas por material

Use capas para organizar por material al ejecutar trabajos mixtos:

```
Material 1 (Acrílico de 3mm):
  - Capa de grabado de acrílico
  - Capa de corte de acrílico

Material 2 (Contrachapado de 3mm):
  - Capa de grabado de madera
  - Capa de corte de madera
```

**Flujo de trabajo:**

1. Procesar todas las capas del Material 1
2. Cambiar materiales
3. Procesar todas las capas del Material 2

**Alternativa:** Use documentos separados para diferentes materiales.

### Pausar entre capas

Puede configurar Rayforge para hacer una pausa entre capas. Esto es útil cuando
necesita:

- Cambiar materiales a mitad del trabajo
- Inspeccionar el progreso antes de continuar
- Ajustar el enfoque para diferentes operaciones

Para configurar las pausas entre capas, use la función de hooks en los ajustes
de su máquina.

### Ajustes específicos por capa

El flujo de trabajo de cada capa puede tener ajustes únicos:

| Capa    | Operación | Velocidad  | Potencia | Pasadas |
| ------- | --------- | ---------- | -------- | ------- |
| Grabado | Ráster    | 300 mm/min | 20%      | 1       |
| Marcado | Contorno  | 500 mm/min | 10%      | 1       |
| Corte   | Contorno  | 100 mm/min | 90%      | 2       |

---

## Mejores prácticas

### Convenciones de nomenclatura

**Buenos nombres de capas:**

- "Grabado - Logo"
- "Corte - Contorno exterior"
- "Marcado - Líneas de pliegue"
- "Pasada 1 - Corte rough"
- "Pasada 2 - Corte final"

**Nombres de capas deficientes:**

- "Capa 1", "Capa 2" (no descriptivos)
- Descripciones largas (mantener conciso)

### Organización de capas

1. **De arriba a abajo = orden de ejecución**
2. **Grabado antes de corte** (regla general)
3. **Agrupar operaciones relacionadas** (todo el corte, todo el grabado)
4. **Usar visibilidad** para enfocarse en el trabajo actual
5. **Eliminar capas no usadas** para mantener los proyectos limpios

### Preparar archivos SVG para importación de capas

**Para mejores resultados al importar capas SVG:**

1. **Use el panel de Capas** en su software de diseño para organizar su diseño
2. **Asigne nombres significativos** a cada capa (ej.: "Grabado", "Corte")
3. **Mantenga las capas planas** - evite poner capas dentro de otras capas
4. **Guarde su archivo** e impórtelo en Rayforge
5. **Verifique la detección de capas** revisando el diálogo de importación

Rayforge funciona mejor con archivos SVG creados en Inkscape o software similar
de diseño vectorial que soporte capas.

### Rendimiento

**Muchas capas:**

- Sin impacto significativo en el rendimiento
- 10-20 capas es común para trabajos complejos
- Organice lógicamente, no para minimizar la cantidad de capas

**Simplifique si es necesario:**

- Combine operaciones similares en una capa cuando sea posible
- Use menos grabados ráster (son los que más recursos consumen)

---

## Solución de problemas

### La capa no genera G-code

**Problema:** La capa aparece en el documento pero no en el G-code generado.

**Soluciones:**

1. **Verifique que la capa tiene piezas de trabajo** - Las capas vacías se
   omiten
2. **Verifique que el flujo de trabajo está configurado** - La capa necesita
   una operación
3. **Verifique los ajustes de operación** - Potencia > 0, velocidad válida,
   etc.
4. **Verifique la visibilidad de las piezas** - Las piezas ocultas pueden no
   procesarse
5. **Regenere el G-code** - Haga un pequeño cambio para forzar la regeneración

### Orden incorrecto de capas

**Problema:** Las operaciones se ejecutan en un orden inesperado.

**Solución:** Reordene las capas en el panel de Capas. Recuerde: arriba =
primero.

### Capas superpuestas en la vista previa

**Problema:** Múltiples capas muestran contenido superpuesto en la vista previa.

**Aclaración:** Esto es normal si las capas comparten la misma área XY.

**Soluciones:**

- Use la visibilidad de capas para ocultar otras capas temporalmente
- Verifique la vista previa 3D para ver la profundidad/orden
- Confirme que esto es intencional (ej.: grabar y luego cortar la misma forma)

### Pieza de trabajo en la capa incorrecta

**Problema:** La pieza de trabajo fue asignada a una capa incorrecta.

**Solución:** Arrastre la pieza de trabajo a la capa correcta en el panel de
Capas o en el árbol de Documento.

### Capas SVG no detectadas

**Problema:** Se importa un archivo SVG pero no aparecen capas en el diálogo de
importación.

**Soluciones:**

1. **Verifique la estructura del SVG** - Abra su archivo en Inkscape o software
   similar para verificar que tiene capas
2. **Active "Usar vectores originales"** - La selección de capas solo está
   disponible en este modo de importación
3. **Verifique que su diseño tiene capas** - Asegúrese de haber creado capas en
   su software de diseño, no solo grupos
4. **Verifique si hay capas anidadas** - Las capas dentro de otras capas pueden
   no detectarse correctamente
5. **Guarde nuevamente su archivo** - A veces volver a guardar con una versión
   actualizada de su software de diseño ayuda

### La importación de capas SVG muestra contenido incorrecto

**Problema:** La capa importada muestra contenido de otras capas o está vacía.

**Soluciones:**

1. **Verifique la selección de capas** - Confirme que las capas correctas están
   activadas en el diálogo de importación
2. **Verifique su diseño** - Abra el archivo original en su software de diseño
   para confirmar que cada capa contiene el contenido correcto
3. **Verifique elementos compartidos** - Elementos que aparecen en múltiples
   capas pueden causar confusión
4. **Pruebe el modo de trazado** - Use el modo de trazado como alternativa si la
   importación de vectores tiene problemas

---

## Páginas relacionadas

- [Operaciones](./operations/contour) - Tipos de operaciones para flujos de
  trabajo de capas
- [Modo de simulación](./simulation-mode) - Vista previa de la ejecución con
  múltiples capas
- [Macros y Hooks](../machine/hooks-macros) - Hooks a nivel de capa para
  automatización
- [Vista previa 3D](../ui/3d-preview) - Visualizar la pila de capas
