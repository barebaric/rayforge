# Configuración de Eje Rotativo

Rayforge soporta accesorios rotativos para grabar y cortar objetos cilíndricos
como vasos, copas, bolígrafos y material redondo. Cuando se conecta un módulo
rotativo, Rayforge envuelve el trabajo alrededor del cilindro y muestra una vista
previa 3D del resultado.

![Ajustes del Módulo Rotativo](/screenshots/machine-rotary-module.png)

## Cuándo necesitas el modo rotativo

Usa el modo rotativo siempre que tu pieza de trabajo sea cilíndrica. Ejemplos
comunes incluyen:

- Grabar logotipos o texto en artículos de bebida
- Cortar patrones en tubos o tuberías
- Marcar objetos cilíndricos como bolígrafos o mangos de herramientas

Sin el modo rotativo, el eje Y mueve la cabeza láser hacia adelante y atrás en una
cama plana. Con el modo rotativo activado, el eje Y controla la rotación del cilindro,
de modo que el diseño se envuelve alrededor de la superficie.

## Configurar un módulo rotativo

Antes de comenzar, conecta físicamente tu módulo rotativo a la máquina según las
instrucciones del fabricante. Normalmente, esto significa conectarlo al puerto del
controlador de pasos del eje Y en lugar del motor normal del eje Y.

En Rayforge, abre **Ajustes → Máquina** y navega a la página **Rotativo** para
configurar tu módulo:

- **Circunferencia**: Mide la distancia alrededor del objeto que deseas grabar.
  Puedes envolver un trozo de papel o cuerda alrededor del cilindro y medir su
  longitud. Esto le indica a Rayforge el tamaño de la superficie cilíndrica para
  que el diseño se escale correctamente.
- **Micropasos por rotación**: Este es el número de pasos que el motor rotativo
  necesita para una rotación completa. Consulta la documentación de tu módulo
  rotativo para encontrar este valor.

## Modo rotativo por capa

Si tu documento tiene varias capas, puedes activar o desactivar el modo rotativo
independientemente para cada capa. Esto es útil cuando deseas combinar trabajo plano
y cilíndrico en un solo proyecto, o cuando tienes diferentes ajustes rotativos para
diferentes partes del trabajo.

Cuando el modo rotativo está activo en una capa, aparece un pequeño icono rotativo
junto a esa capa en la lista de capas, para que puedas ver de un vistazo qué capas
se ejecutarán en modo rotativo.

## Vista previa 3D en modo rotativo

Cuando el modo rotativo está activo, la [vista 3D](../ui/3d-preview) muestra tu
trayectoria de herramienta envuelta alrededor de un cilindro en lugar de en una
superficie plana.

![Vista previa 3D en modo rotativo](/screenshots/main-3d-rotary.png)

Esto te da una vista previa realista de cómo se verá el diseño en el objeto real,
facilitando la detección de problemas de tamaño o colocación antes de comenzar a
cortar.

## Consejos para buenos resultados

- **Mide la circunferencia con cuidado** — incluso un pequeño error aquí deformará
  tu diseño alrededor del cilindro.
- **Asegura la pieza de trabajo** — asegúrate de que el objeto esté firmemente
  colocado en los rodillos y no tambalee ni se deslice durante el trabajo.
- **Prueba primero con potencia baja** — realiza una pasada de grabado ligera
  para verificar la alineación antes de comprometerte con un corte a potencia
  completa.
- **Mantén la superficie limpia** — el polvo o residuos en el cilindro pueden
  afectar la calidad del grabado.

## Páginas relacionadas

- [Flujo de trabajo multicapa](../features/multi-layer) - Ajustes por capa incluyendo rotativo
- [Vista 3D](../ui/3d-preview) - Vista previa de trayectorias en 3D
- [Ajustes de máquina](general) - Configuración general de máquina
