---
description:
  "Corta vinilo, cartón, cuero y otros materiales laminares con una cuchilla de arrastre o una
  cuchilla tangencial motorizada en lugar de un láser."
---

# Corte con cuchilla

Rayforge también puede controlar portacuchillas además de láseres. Dos pasos integrados cortan
contornos vectoriales con una cuchilla: **Corte con cuchilla de arrastre** para un portacuchillas de
arrastre y **Corte con cuchilla tangencial** para una hoja giratoria motorizada. Los pasos de
cuchilla solo aparecen en máquinas con un cabezal de cuchilla compatible, de modo que una máquina
puede mezclar operaciones láser y de cuchilla en el mismo proyecto.

## Requisitos

- Una máquina con un cabezal **Cuchilla de arrastre** o **Cuchilla tangencial**. Añade el cabezal en
  [Ajustes de la máquina → Cabezales](../machine/general.md) con **Añadir cabezal**.
- Un controlador G-code capaz de mover la máquina sin láser. Los pasos de cuchilla cortan sobre la
  línea central y no usan potencia láser.
- El complemento integrado Drag Knife. Se distribuye con Rayforge; si se desactivó, actívalo en
  **Ajustes → Complementos**.

## Configurar un cabezal de cuchilla

1. Abre **Máquina → Ajustes actuales de la máquina → Cabezales**.
2. Pulsa **Añadir cabezal** y elige **Cuchilla de arrastre** o **Cuchilla tangencial**.
3. Ponle un nombre al cabezal. En una cuchilla de arrastre, introduce el **Desplazamiento de la
   hoja**: la distancia entre la punta de la hoja y el punto de giro de la máquina.
4. Cierra los ajustes. Los pasos de cuchilla correspondientes aparecen ahora al añadir una
   operación.

El cabezal de cuchilla también es una capacidad de la máquina: las operaciones que necesitan una
cuchilla solo están disponibles mientras la máquina activa tiene uno.

## Corte con cuchilla de arrastre

Una cuchilla de arrastre tiene una hoja que va detrás del punto de giro. Rayforge compensa el
desplazamiento automáticamente, de modo que la hoja sigue el contorno en lugar del punto de giro.

1. Selecciona los contornos que quieres cortar.
2. Añade una operación **Corte con cuchilla de arrastre**.
3. Elige el cabezal **Cuchilla** y ajusta **Desplazamiento de la hoja**, **Altura Z segura**,
   **Ángulo de giro** y la velocidad de corte.

El paso reescribe cada contorno en la trayectoria que debe seguir la máquina: un arco de entrada
alinea la hoja con la primera dirección de corte, los segmentos rectos se desplazan hacia delante
por el desplazamiento de la hoja, las esquinas dentro del ángulo de giro se giran con la hoja abajo,
las esquinas más agudas levantan la hoja para girar en el aire y un arco de salida corta más allá
del final de la trayectoria.

Ajusta el **Desplazamiento de la hoja** según la documentación del portacuchillas o mídelo. Un valor
demasiado pequeño deja esquinas redondeadas; uno demasiado grande se pasa de ellas. El **Ángulo de
giro** limita cuánto gira la hoja estando abajo; las esquinas más agudas levantan la cuchilla
primero para que la hoja no arrastre de lado.

## Corte con cuchilla tangencial

Una cuchilla tangencial tiene un motor que gira la hoja para mantenerla tangente a la trayectoria.
El paso añade la dirección de la trayectoria como valor del eje **A** a cada movimiento de corte,
para que el controlador pueda girar la hoja.

1. Selecciona los contornos que quieres cortar.
2. Añade una operación **Corte con cuchilla tangencial**.
3. Elige el cabezal **Cuchilla** y ajusta **Tolerancia de ángulo**, **Tolerancia de radio**, **RPM
   del husillo**, **Altura Z segura** y la velocidad de corte.

Los cambios de dirección mayores que la **Tolerancia de ángulo** o los arcos más cerrados que la
**Tolerancia de radio** levantan la cuchilla a la Z segura, la giran y vuelven a bajar. Los cambios
menores giran la hoja en el sitio con la cuchilla abajo. Las **RPM del husillo** se emiten como un
comando M3 para el motor de la hoja; Rayforge nunca usa M4, porque el motor de la hoja solo gira en
sentido horario.

## Consejos

- Ejecuta el trabajo primero a velocidad reducida, sobre todo con material grueso o fibroso.
- Usa una base de corte sacrificial o una profundidad de corte pequeña para que la hoja no marque la
  cama.
- Los pasos de cuchilla son operaciones G-code normales: se pueden combinar con pasos láser,
  reordenar, duplicar y guardar en recetas como cualquier otra operación.
