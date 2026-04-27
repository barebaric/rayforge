# Deepnest

Deepnest organiza automáticamente tus workpieces en un diseño compacto sobre
tu material de stock o área de trabajo de la máquina. Utiliza un algoritmo
genético para encontrar un empaquetado eficiente de formas, minimizando el
desperdicio y ajustando más piezas en cada lámina.

![Diálogo de configuración de Deepnest](/screenshots/addon-deepnest.png)

## Requisitos previos

Selecciona uno o más workpieces en el lienzo antes de ejecutar el nesting.
También puedes seleccionar elementos de stock para definir los límites de la
lámina. Si no se selecciona ningún stock, el addon utiliza el stock del
documento o recurre al área de trabajo de la máquina.

## Ejecutar el diseño de nesting

Inicia el diseño de nesting desde el menú **Organizar**, el botón de la
barra de herramientas o el atajo de teclado **Ctrl+Alt+N**. Se abre un
diálogo de configuración antes de que el algoritmo se ejecute.

## Configuración de nesting

El diálogo de configuración ofrece las siguientes opciones antes de que
comience el algoritmo de nesting.

**Espaciado** establece la distancia entre las formas anidadas, en
milímetros. El valor predeterminado se toma del tamaño del spot láser de tu
máquina. Aumenta este valor para añadir un margen de seguridad entre las
piezas.

**Restringir rotación** mantiene todas las piezas en su orientación original.
Cuando esta opción está desactivada, el algoritmo rota las piezas en
incrementos de 10 grados para encontrar un ajuste más estrecho. Dejar la
rotación sin restringir produce un mejor uso del material pero tarda más en
calcularse.

**Permitir volteo horizontal** espeja las piezas horizontalmente durante el
nesting. Esto puede ayudar a ajustar las piezas más estrechamente, pero los
cortes resultantes estarán espejados.

**Permitir volteo vertical** espeja las piezas verticalmente durante el
nesting. La misma consideración sobre la salida espejada se aplica aquí.

Haz clic en **Iniciar nesting** para comenzar. El diálogo se cierra y el
algoritmo se ejecuta en segundo plano. Aparece un indicador de progreso en el
panel inferior mientras el nesting está en curso.

## Después del nesting

Cuando el algoritmo termina, todos los workpieces en el lienzo se
reposicionan a sus ubicaciones anidadas. Las posiciones se aplican como una
sola acción deshacible, por lo que puedes deshacer el diseño con un paso si
el resultado no es lo que necesitas.

Si el algoritmo no pudo colocar todos los workpieces en el stock disponible,
los elementos no colocados se mueven a la derecha del área de stock para que
permanezcan visibles y sean fáciles de identificar.

Si el resultado del nesting es peor que el diseño original — por ejemplo, si
las piezas ya encajaban bien — los workpieces permanecen en sus posiciones
originales.

## Temas relacionados

- [Manejo de stock](../features/stock-handling) - Definir material de stock para el nesting
- [Posicionamiento de workpieces](../features/workpiece-positioning) - Posicionar workpieces manualmente
