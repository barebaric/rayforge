---
description:
  "Corta líneas discontinuas en Rayforge con el posprocesador Perforación: define una longitud de
  corte y una longitud de salto para pliegues, líneas de rasgado y agujeros de costura."
---

# Perforación

No todos los cortes sirven para separar una pieza del material. Hay líneas que existen para que el
material se pliegue sobre ellas, se rasgue limpiamente o marque el camino de la aguja. Para eso está
el posprocesador **Perforación**: convierte un corte continuo en uno discontinuo. En lugar de
mantener el láser encendido de principio a fin, dispara durante la **Longitud de corte**, recorre la
**Longitud de salto** con el láser apagado, y repite ese ritmo a lo largo de cada contorno de la
pieza. Lo que queda es una línea de cortes pequeños que mantiene unido el material — hasta que
quieras que ceda.

![Los ajustes de Perforación en la página de posprocesado de un paso de Contorno](/screenshots/step-settings-contour-post-perforation.webp)

## Algunos usos típicos

Allí donde una línea deba debilitar el material sin cortarlo por completo, vale la pena un corte
perforado. Los casos clásicos:

- **Líneas de pliegue** en cartulina y cartón ondulado se pliegan con precisión sobre un corte
  discontinuo: cajas, tarjetas y diseños pop-up, por ejemplo.
- **Líneas de rasgado** abren entradas, cupones y envases sin necesidad de tijeras.
- **Agujeros de costura** en cuero marcan el camino de la aguja antes de coser.
- **Contornos discontinuos** son además un efecto decorativo en sí mismos, sobre madera, papel o
  acrílico.

## Activar la perforación

La perforación se encuentra en la página **Posprocesado** de los ajustes de un paso de
[Contorno](operations/contour), junto a los demás posprocesadores. Viene desactivada por defecto; al
activar el interruptor aparecen dos valores:

- La **Longitud de corte** es cuánto permanece encendido el láser antes de cada hueco.
- La **Longitud de salto** es la distancia recorrida con el láser apagado entre dos cortes.

El patrón empieza al inicio de cada contorno con un corte completo y vuelve a empezar en cada
contorno, de modo que todas las líneas discontinuas de la pieza comienzan igual. Un contorno
demasiado corto para un solo corte y un salto simplemente se corta entero.

## El resultado

En el lienzo, la línea discontinua sustituye al contorno de corte sólido: cada hueco del trazado es
también un hueco en la pieza terminada.

![Un rectángulo y un círculo se cortan como líneas discontinuas en una plancha de madera
contrachapada](/screenshots/operations-perforation.webp)

## Consejos

Para una línea de pliegue que no debe atravesar el material, empieza con una longitud de salto
parecida a la de corte y baja la potencia: un surco poco profundo pliega igual de bien que un corte
discontinuo profundo, y el reverso queda limpio.

Las longitudes muy cortas (unas décimas de milímetro) hacen que el láser se encienda y apague a toda
velocidad, lo que reduce la potencia efectiva y puede dejar marcas irregulares. Si el patrón se ve
desigual, prueba con longitudes mayores.

La perforación se combina bien con el resto del posprocesado: las
[Pestañas de Sujeción](holding-tabs), la [Entrada / Salida](lead-in-out) y la
[Multi-Pasada](multi-pass) siguen funcionando como siempre.

## Páginas relacionadas

- [Corte de Contorno](operations/contour) — la operación de corte a la que pertenece este
  posprocesador
- [Pestañas de Sujeción](holding-tabs) — huecos colocados a mano para mantener las piezas unidas
