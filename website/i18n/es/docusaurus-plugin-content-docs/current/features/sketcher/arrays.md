---
description: "Cree arreglos circulares y arreglos a lo largo de una curva en el diseñador paramétrico de Rayforge."
---

# Arreglos

El diseñador proporciona dos herramientas de arreglo para crear arreglos
paramétricos: **Arreglo Circular** y **Arreglo a lo largo de curva**.

## Arreglos Circulares

La herramienta de **Arreglo Circular** (`G+Y`) crea un patrón polar paramétrico
a partir de la selección actual:

1. Seleccione las entidades que desea distribuir en patrón.
2. Active la herramienta desde la barra de herramientas, el menú
   **Boceto → Arreglos**, o con `G+Y`.
3. Aparece una guía circular en el lienzo y se abre un diálogo no modal
   con una vista previa en vivo.
4. Establezca el **conteo** y el **ángulo total**. Las copias se generan
   paramétricamente alrededor del centro de la guía circular.
5. Arrastre el centro de la guía circular para reposicionar el arreglo, o
   arrastre la entidad original para cambiar el radio — los campos del
   diálogo se actualizan en vivo.
6. El **radio** de la guía circular redimensiona todo el arreglo. Haga
   **doble clic** en la guía circular para reabrir el diálogo de edición y
   regenerar miembros faltantes o redistribuir.

Las copias son geometría estática horneada sin restricciones del
resolvedor: se regeneran a partir de la plantilla cuando se edita el
arreglo. Eliminar un miembro solo elimina la geometría de ese miembro y
nunca redistribuye los supervivientes.

## Arreglo a lo largo de curva

La herramienta **Arreglo a lo largo de curva** distribuye copias de una o más
entidades a lo largo de una guía de ruta (una línea, arco o curva Bézier). Las
copias se colocan directamente en la ruta y siguen su tangente en cada posición.

### Crear un arreglo a lo largo de curva

1. Dibuje la forma que desea distribuir (la semilla) y la guía de ruta que
   desea seguir.
2. Seleccione ambas: primero haga clic en la **guía de ruta**, luego
   Mayúsculas-clic en las **entidades semilla**.
3. Active la herramienta desde la barra de herramientas, el menú
   **Boceto → Arreglos**, o `G+W`.
4. Se abre un diálogo no modal mostrando una vista previa en vivo con copias
   distribuidas a lo largo de la ruta.
5. Ajuste el **conteo** (total de miembros incluyendo la plantilla al inicio
   de la ruta) o establezca un valor de **espaciado** para derivar el conteo
   automáticamente de la longitud de la ruta.
6. Opcionalmente habilite **Alinear a la tangente** para que cada copia rote
   para seguir la dirección de la ruta en su posición.
7. Use **Desplazamiento desde el inicio** para saltar una sección principal
   de la ruta antes de colocar la primera copia.

### Editar un arreglo a lo largo de curva

- **Haga doble clic** en la guía de ruta (o haga clic en **Editar** en la
  barra de herramientas) para reabrir el diálogo y cambiar el conteo,
  espaciado, desplazamiento o configuraciones de alineación.
- **Arrastre** cualquier extremo de la guía de ruta para remodelarla. Cuando
  suelte, todas las copias se redistribuyen automáticamente a lo largo de la
  nueva geometría de la ruta — incluyendo actualizaciones de rotación cuando
  *Alinear a la tangente* está habilitado.
- La forma semilla se puede editar como cualquier otra geometría del boceto;
  los cambios se propagan a todas las copias en la próxima actualización.

### Cómo funciona

Las copias son geometría estática horneada — no están vinculadas a la
plantilla a través de restricciones del resolvedor. Cuando se edita la
guía de ruta, `sync_arrays` detecta el cambio y regenera todas las
copias desde cero usando la geometría actual de la ruta. Esto mantiene
las actualizaciones rápidas y evita la sobrecarga del resolvedor.

La plantilla (ranura 0) se coloca al inicio de la ruta. Su posición y
orientación se actualizan automáticamente cuando se edita la ruta. Las
entidades semilla originales se eliminan cuando se crea el arreglo;
deshacer las restaura.
