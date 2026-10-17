---
description:
  "Graba códigos QR, Code 128 y códigos de barras EAN-13 en Rayforge. El contenido puede ser una
  plantilla, para que cada pieza lleve su propio código."
---

# Códigos de barras

El complemento Códigos de barras añade códigos QR y códigos de barras como piezas de trabajo. Se
dibujan como contornos vectoriales con relleno, de modo que una operación de Grabado los graba
limpiamente y una operación de Contorno aún puede seguirlos.

## Añadir un código de barras

Elige **Objeto - Añadir código de barras**. Aparece un código QR en el centro del área de trabajo,
ya seleccionado. Edítalo en el grupo **Código de barras** del panel de propiedades:

- **Contenido** — el texto a codificar. Pulsa `Enter` o el botón de aplicar para actualizar el
  código.
- **Tipo** — Código QR, Code 128 o EAN-13.
- **Corrección de errores** (QR) — cuánto del código puede estar dañado y seguir leyéndose: Baja,
  Media, Cuartil o Alta. Los niveles más altos hacen el código más grande.
- **Tamaño de módulo** — el ancho de la barra más estrecha o de un cuadrado QR. El tamaño de módulo
  y la altura de barra se muestran en la unidad de longitud que hayas elegido. La pieza toma el
  tamaño del código, así que este es el ajuste que determina su tamaño.
- **Altura de barra** (Code 128, EAN-13) — la altura de las barras.
- **Mostrar texto** (Code 128, EAN-13) — imprime el contenido debajo de las barras. EAN-13 usa su
  disposición habitual, con las barras de guarda bajando entre los dígitos.
- **Invertir** — graba las partes claras en su lugar, para material oscuro: el código queda claro
  sobre un borde oscuro. **Zona de silencio** fija ese borde en módulos y solo se puede cambiar
  mientras **Invertir** está activado; por defecto es el mínimo que necesita el tipo.

Deja un margen libre alrededor de un código no invertido: 4 módulos para códigos QR, 10 para Code
128 y 9 para EAN-13.

## Tipos

- **Código QR** admite cualquier texto, incluidos enlaces.
- **Code 128** admite ASCII imprimible. Las secuencias de dígitos se empaquetan por pares
  automáticamente.
- **EAN-13** toma 12 dígitos y añade el dígito de control. Con 13 dígitos, el dígito de control se
  verifica. El campo de contenido se vuelve rojo cuando el contenido no encaja con el tipo; pasa el
  ratón por encima para ver el motivo.

## Plantillas

El contenido puede contener expresiones `{...}` como un cuadro de texto, por ejemplo `{2 * 21}`. Con
[Texto variable](../features/variable-text.md), `SN-{serial:05d}` da a cada pieza su propio número
de serie, y las copias de una matriz siguen contando.
