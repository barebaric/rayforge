# Smart Stock

Smart Stock utiliza visión por computadora para detectar el material colocado en
la cama del láser y crear elementos de stock coincidentes en tu documento.
Comparando una imagen de referencia de la cama vacía con la vista actual de la
cámara, el addon identifica los contornos del stock físico y genera elementos
de stock correctamente posicionados con la forma y tamaño correctos.

## Requisitos previos

Necesitas una cámara configurada y calibrada conectada a tu máquina. La cámara
debe estar configurada con corrección de perspectiva para que la imagen capturada
se alinee con el sistema de coordenadas físico de la máquina. También necesitas
una máquina configurada para que el addon conozca las dimensiones del área de
trabajo.

## Abrir el diálogo de detección

Abre el diálogo desde **Herramientas - Detectar Stock desde Cámara**. La
ventana muestra una vista previa de la cámara en vivo a la izquierda y la
configuración de detección a la derecha.

## Capturar una imagen de referencia

Antes de detectar stock, necesitas una imagen de referencia de la cama del láser
vacía. Sin material sobre la cama, haz clic en el botón **Capturar** junto a
**Capturar Referencia**. El addon almacena esta imagen y la compara con la
transmisión de la cámara en vivo para encontrar nuevos objetos.

Las imágenes de referencia se guardan por cámara. Cuando vuelvas a abrir el
diálogo con la misma cámara, la referencia capturada anteriormente se carga
automáticamente y la detección se ejecuta inmediatamente si ya hay material
sobre la cama.

## Detectar stock

Coloca tu material sobre la cama del láser y haz clic en **Detectar Stock** en
la parte inferior del panel de configuración. El addon compara el fotograma
actual de la cámara con la imagen de referencia y traza los contornos de
cualquier objeto nuevo. Las formas detectadas aparecen en la vista previa como
contornos magenta con relleno verde.

La fila de estado en la parte inferior del panel de configuración informa
cuántos elementos se encontraron. Si no se detecta stock, ajusta la colocación
o la iluminación e inténtalo de nuevo.

## Configuración de detección

**Cámara** muestra la cámara seleccionada actualmente. Haz clic en **Cambiar**
para cambiar a una cámara configurada diferente.

**Sensibilidad** controla cuánto cambio visual se requiere para registrar como
stock. Con valores más altos, se detectan diferencias más pequeñas o sutiles
entre la referencia y el fotograma actual. Con valores más bajos, solo se
detectan cambios grandes. Si el addon no detecta material que está presente,
aumenta la sensibilidad. Si detecta sombras o reflejos como stock, redúcela.

**Suavizado** controla la suavidad de los contornos detectados. Valores más
altos producen contornos más redondeados y simples al filtrar los pequeños
bordes irregulares de la imagen de la cámara. Valores más bajos conservan más
detalle de la forma real del material.

## Crear elementos de stock

Una vez que la vista previa muestre los contornos detectados coincidiendo con tu
material, haz clic en **Crear Elementos de Stock** en la barra de título. El
addon añade un activo de stock y un elemento de stock a tu documento por cada
forma detectada, posicionados en las coordenadas físicas correctas en el lienzo.
El diálogo se cierra después de crear los elementos.

## Temas relacionados

- [Configuración de cámara](../machine/camera) - Configurar y calibrar tu cámara
- [Manejo de stock](../features/stock-handling) - Trabajar con elementos de stock en tu documento
