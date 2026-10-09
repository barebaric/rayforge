---
description:
  "Corta líneas discontinuas en Rayforge con el posprocesador Perforación: define una longitud de
  corte y una longitud de salto para pliegues, líneas de rasgado y agujeros de costura."
---

# Perforación

El posprocesador **Perforación** convierte un corte continuo en uno discontinuo. El láser dispara
durante la **Longitud de corte**, luego se desplaza con el láser apagado durante la **Longitud de
salto**, y repite esto a lo largo de cada contorno de la pieza.

## Cuándo usarlo

- Líneas de pliegue en cartulina y cartón ondulado
- Líneas de rasgado (entradas, cupones, embalajes)
- Agujeros de costura en cuero
- Líneas discontinuas decorativas

## Ajustes

La perforación está disponible en los ajustes de posprocesado de las operaciones de **Contorno**.
Está desactivada por defecto.

- **Longitud de corte**: la distancia que dispara el láser antes de cada hueco.
- **Longitud de salto**: la distancia recorrida con el láser apagado entre dos cortes.

El patrón se mide a lo largo de cada contorno de la pieza y vuelve a empezar al inicio de cada
contorno, siempre con un corte completo. Un contorno más corto que un corte más un salto se corta
entero.

## Consejos

- Para una línea de pliegue que no debe atravesar el material, empieza con una longitud de salto
  similar a la de corte y baja la potencia.
- Longitudes muy cortas (unas décimas de milímetro) hacen que el láser se encienda y apague
  rápidamente, lo que reduce la potencia efectiva.
- La perforación funciona junto con [Pestañas de sujeción](holding-tabs),
  [Entrada/Salida](lead-in-out) y [Multipasada](multi-pass).

## Páginas relacionadas

- [Corte de contorno](operations/contour) - La operación de corte que usa la perforación
- [Pestañas de sujeción](holding-tabs) - Huecos colocados a mano para mantener las piezas unidas
