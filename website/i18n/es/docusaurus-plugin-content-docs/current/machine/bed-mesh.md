---
description:
  "Sondea la superficie de trabajo de tu láser en una cuadrícula y compensa automáticamente una cama
  irregular. Mantiene el punto focal sobre el material en camas grandes u onduladas."
---

# Malla de la cama

Las superficies de trabajo grandes rara vez son perfectamente planas. Cuando la altura de la cama
varía más que la profundidad de enfoque de tu láser, los cortes resultan inconsistentes a lo largo
del material. La función de malla de la cama mide la altura de la superficie en una cuadrícula y
compensa el eje Z del trayecto para que el punto de enfoque siga la superficie real.

![Malla de la cama](/screenshots/machine-settings-bed-mesh.webp)

## Requisitos

La malla de la cama necesita una máquina con eje Z y un controlador que admita sondeo (GRBL, Marlin,
Smoothie y OctoPrint actualmente). La página **Malla de la cama** solo aparece en **Ajustes →
Máquina** cuando se cumplen ambas condiciones. Está situada después de la página de Dispositivo.

## Sondar la cama

Abre **Ajustes → Máquina** y navega a la página **Malla de la cama**.

1. **Cuadrícula de sondeo**: Define el área a sondar (origen X/Y, ancho, alto) y la densidad de la
   cuadrícula (columnas y filas). La página muestra el número de puntos de sondeo resultante y
   estima la duración. Las cuadrículas más densas siguen la superficie con más precisión, pero
   tardan más.
2. **Sondeo**: Configura la velocidad de sondeo, cuánto puede bajar la cabeza buscando la superficie
   en cada punto (Recorrido máximo) y la altura Z segura para moverse entre puntos.
3. Pulsa **Iniciar sondeo**. La máquina visita cada punto de la cuadrícula en patrón serpenteante y
   toca la superficie en cada uno; la vista 3D se rellena en vivo a medida que llegan los
   resultados. Puedes detener la ejecución en cualquier momento; la malla solo se guarda cuando la
   cuadrícula completa termina.

La malla se guarda con el perfil de la máquina y se muestra como una superficie 3D coloreada: las
zonas azules son más bajas y las rojas más altas. Usa el botón **Eliminar malla** para borrarla si
ya no quieres compensación de altura.

:::note Durante el sondeo la cabeza se mueve por toda el área de la cuadrícula. Despeja la cama de
objetos que puedan bloquear el sensor y asegúrate de que tu punta de sondeo (o la retícula del
láser) pueda alcanzar la superficie en cada punto de la cuadrícula. :::

## Aplicar la malla

Sondear solo registra el mapa de altura; aplicarlo se controla por trabajo:

- **Aplicar a los trabajos por defecto**: Activa este interruptor en la página Malla de la cama para
  compensar el eje Z de cada trabajo en esta máquina.
- **Anulación por capa**: En los ajustes de una capa (Ajustes de capa → Posprocesamiento), el valor
  por defecto de la máquina aparece con las acciones **Personalizar** y **Desactivar**. Personalizar
  copia la corrección a la capa, donde puedes ajustar el desfase Z (p. ej. para enfocar ligeramente
  por encima de la superficie); desactivar apaga la corrección solo para esa capa. Al eliminar la
  entrada de la capa se vuelve al valor por defecto de la máquina.

La corrección se aplica por igual a los movimientos de corte y de desplazamiento, y los arcos siguen
siendo arcos (se vuelven helicoidales). Se ejecuta sobre el trayecto final en coordenadas de
máquina, por lo que no le afectan los sistemas de coordenadas de trabajo ni la esquina de origen de
la máquina.

---

## Páginas relacionadas## Páginas relacionadas

- [Ajustes de hardware](hardware) - Dimensiones de la máquina y configuración de ejes
- [Ajustes del dispositivo](device) - Conexión y opciones del controlador
- [Vista 3D](../ui/3d-preview.md) - Visualización 3D del trayecto
