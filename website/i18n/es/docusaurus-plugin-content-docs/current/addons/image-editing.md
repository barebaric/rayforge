# Edición de imágenes

El complemento de edición de imágenes cambia cómo se ve la imagen de una pieza sin tocar el archivo
importado. Cada edición se guarda con la pieza, de modo que puedes deshacerla, quitarla más tarde o
colocar el mismo archivo varias veces con ediciones diferentes.

El lienzo, la vista previa 3D y cada operación usan la imagen editada. Lo que ves en el lienzo es lo
que el láser graba.

## Invertir imagen

Selecciona una o más piezas y elige **Tools - Invert Image**. Las áreas claras se vuelven oscuras y
las áreas oscuras se vuelven claras; las áreas transparentes permanecen transparentes, por lo que el
fondo sigue sin grabarse.

![Una foto en escala de grises de montañas y un lago, tal como se importó](/images/image-edit-invert-before.webp)

![La misma foto después de Invert Image](/images/image-edit-invert-after.webp)

Elígelo de nuevo para quitar la inversión. Con una selección mixta, las piezas que aún no están
invertidas se invierten, de modo que toda la selección acaba igual.

Úsalo para litofanías y repujado en una operación de grabado, donde las áreas claras deben grabarse
en profundidad, o para grabar dibujos claros sobre un fondo oscuro.

## Proyectos antiguos

Antes de este complemento, Invertir era un ajuste de la operación de grabado. Cuando abres un
proyecto antiguo que lo usaba, Rayforge mueve el ajuste a las piezas de esa capa, para que el
resultado sea el mismo.

## Cuando el complemento está desactivado

Un proyecto conserva sus ediciones de imagen cuando el complemento está desactivado o no está
instalado. Rayforge te avisa de que el proyecto usa funciones no disponibles y muestra las imágenes
sin esas ediciones hasta que el complemento vuelva a estar activado.
