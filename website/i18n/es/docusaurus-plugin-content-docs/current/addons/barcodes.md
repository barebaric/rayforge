---
description:
  "Graba códigos QR, Code 128 y EAN-13 en Rayforge. Los códigos son activos reutilizables cuyo
  contenido puede ser una plantilla, para que cada pieza lleve su propio código."
---

# Códigos de barras

El complemento Códigos de barras añade códigos QR, Code 128 y EAN-13 como activos reutilizables. Se
dibujan como contornos vectoriales con relleno, de modo que una operación de Grabado los graba
limpiamente y una de Contorno aún puede seguirlos.

## Añadir un código de barras

![El grupo Código de barras con un símbolo Code 128](/screenshots/addon-barcodes.webp)

Elige **Objeto - Añadir código de barras**. Aparece un código QR en el centro del área de trabajo,
ya seleccionado. Edítalo en el grupo **Código de barras** del panel de propiedades:

- **Nombre** — el nombre del activo en el explorador de activos.
- **Contenido** — el texto a codificar. Pulsa `Enter` o el botón de aplicar.
- **Tipo** — Código QR, Code 128 o EAN-13.
- **Corrección de errores** (QR) — cuánto del código puede dañarse: Baja, Media, Cuartil o Alta. Los
  niveles más altos hacen el código más grande.
- **Tamaño de módulo** — el ancho de la barra más estrecha o de un cuadrado QR. El tamaño de módulo
  y la altura de barra siguen tu unidad de longitud.
- **Altura de barra** (Code 128, EAN-13) — la altura de las barras.
- **Mostrar texto** (Code 128, EAN-13) — imprime el contenido bajo las barras. EAN-13 usa su
  disposición habitual, con las barras de guarda bajando entre los dígitos.
- **Invertir** — graba las partes claras, para material oscuro. La **zona de silencio** fija el
  borde oscuro en módulos y solo se puede cambiar con Invertir activado.

## El explorador de activos

Cada código también es un activo. Abre el explorador de activos en el panel inferior:

- **Añadir** (`+`) crea un activo sin colocarlo. Usa **Crear nueva pieza**, arrástralo al lienzo o
  haz doble clic en la tarjeta.
- Hacer doble clic en una pieza con código la selecciona y muestra sus propiedades.
- Duplicar, copiar, cortar y eliminar funcionan como con otros activos.

Cambiar el activo actualiza todas las copias colocadas. Usa plantillas cuando las copias deban
diferir.

Deja un margen libre alrededor de un código no invertido: 4 módulos para QR, 10 para Code 128 y 9
para EAN-13.

## Tipos

- **Código QR** admite cualquier texto, incluidos enlaces.
- **Code 128** admite ASCII imprimible. Las secuencias de dígitos se empaquetan por pares.
- **EAN-13** toma 12 dígitos y añade el dígito de control. Con 13 dígitos se verifica. El campo se
  vuelve rojo cuando el contenido no encaja; pasa el ratón para ver el motivo.

## Plantillas

El contenido puede incluir expresiones `{...}` como un cuadro de texto, por ejemplo `{2 * 21}`. Con
el [contador de trabajos](../features/job-counter.md), `SN-{counter:05d}` da a cada pieza su propio
número, y las copias de matriz siguen contando.
