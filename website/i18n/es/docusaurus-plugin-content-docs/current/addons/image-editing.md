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

## Ajustar imagen

Selecciona una o más piezas de trabajo y elige **Tools - Adjust Image...** para corregir una foto
antes de grabarla o vectorizarla:

- **Brightness** (brillo) aclara u oscurece la imagen.
- **Contrast** (contraste) aleja los tonos del gris medio o los acerca a él.
- **Gamma** por encima de 1 aclara los tonos medios y por debajo de 1 los oscurece; el negro y el
  blanco no cambian.
- **Sharpen** (enfocar) refuerza los bordes y los detalles finos (una máscara de enfoque). **Sharpen
  Radius** fija el tamaño de esos detalles en milímetros, de modo que el resultado es el mismo con
  cualquier intervalo de línea o nivel de zoom.

![El diálogo de Ajustar imagen con los controles deslizantes de tono y nitidez](/screenshots/addon-image-editing-adjust.webp)

El lienzo muestra los cambios mientras editas. **Apply** los conserva como un único paso de
deshacer; si cierras el diálogo sin aplicar, se descartan, y **Reset** devuelve todos los valores a
neutro. Las zonas transparentes siguen siendo transparentes.

El diálogo se abre con los valores de la primera pieza seleccionada, y Apply da los mismos valores a
todas las piezas seleccionadas. El enfoque se aplica antes de las correcciones de tono, y otras
ediciones como Invert Image mantienen su lugar.

## Proyectos antiguos

Antes de este complemento, Invertir era un ajuste de la operación de grabado. Cuando abres un
proyecto antiguo que lo usaba, Rayforge mueve el ajuste a las piezas de esa capa, para que el
resultado sea el mismo.

## Cuando el complemento está desactivado

Un proyecto conserva sus ediciones de imagen cuando el complemento está desactivado o no está
instalado. Rayforge te avisa de que el proyecto usa funciones no disponibles y muestra las imágenes
sin esas ediciones hasta que el complemento vuelva a estar activado.
