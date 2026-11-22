---
description:
  "Dibuje polígonos regulares y estrellas en el diseñador de Rayforge. Están formados por puntos y
  líneas normales que las restricciones mantienen regulares."
---

# Polígono y estrella

La herramienta **polígono** (`G+X`) dibuja un polígono regular, como un hexágono para un posavasos,
o una estrella.

![Un hexágono y una estrella de cinco puntas](/screenshots/addons-sketcher-tool-polygon.webp)

## Dibujar un polígono

Haga clic donde debe estar el centro, mueva el puntero y vuelva a hacer clic, o pulse en el centro,
arrastre y suelte. El puntero fija la primera esquina, por lo que determina tanto el tamaño como la
rotación. Un polígono nuevo tiene 6 esquinas.

- Mantenga `Ctrl` para dibujar una estrella en su lugar: añade una esquina interior entre cada dos
  esquinas exteriores, a la mitad del radio exterior.

Mientras la vista previa está activa, puede introducir valores exactos: la barra de estado muestra
los campos `R`, `N` y `Star`. `R` es el radio desde el centro hasta las esquinas, `N` el número de
esquinas (de 3 a 100) y `Star` el radio interior como fracción del exterior (entre 0 y 1).
Introducir un valor en `Star` convierte el polígono en una estrella. Escriba un valor, pulse `Tab`
para pasar al siguiente campo y `Enter` para crear la forma; `Esc` cancela la vista previa. La
herramienta recuerda el número de esquinas para el siguiente polígono.

## Qué obtiene

Un polígono no es un objeto especial: consta de puntos y líneas normales, por lo que el redondeo, el
chaflán, el desplazamiento y las operaciones booleanas funcionan con él como con cualquier otra
forma.

Las restricciones lo mantienen regular. Todos los lados tienen la misma longitud y las esquinas
están sobre un círculo de construcción alrededor del punto central; una estrella tiene un segundo
círculo de construcción para sus esquinas interiores y mantiene la proporción entre ambos radios. Si
introdujo un radio, una restricción de radio lo fija. Después puede arrastrar el centro para mover
el polígono, o arrastrar una esquina para escalarlo y girarlo, y seguirá siendo regular. Para
cambiar el tamaño más adelante, edite la restricción de radio o añada una al círculo de
construcción.
