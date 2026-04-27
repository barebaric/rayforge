# Print & Cut

Alinea los cortes láser sobre material preimpreso registrando puntos de
referencia en tu diseño y asignándolos a sus posiciones físicas en el material.
Esto es útil para cortar pegatinas, etiquetas o cualquier elemento que deba
coincidir con una impresión existente.

## Requisitos previos

El addon requiere una máquina configurada. Tu máquina debe estar conectada para
el paso de desplazamiento. También necesitas un workpiece o grupo seleccionado
en el lienzo.

## Abrir el asistente

Selecciona un solo workpiece o grupo en el lienzo y luego abre
**Herramientas - Alinear a posición física**. El asistente se abre como un
diálogo de tres pasos con una vista previa de tu workpiece a la izquierda y
los controles a la derecha.

## Paso 1: Seleccionar puntos de diseño

![Seleccionar puntos de diseño](/screenshots/addon-print-and-cut-pick.png)

El panel izquierdo muestra una representación de tu workpiece seleccionado.
Haz clic directamente sobre la imagen para colocar el primer punto de
alineación, marcado en verde, luego haz clic de nuevo para colocar el segundo
punto, marcado en azul. Una línea discontinua conecta los dos puntos.

Elige dos puntos que correspondan a características identificables en tu
material físico — por ejemplo, marcas de registro impresas o esquinas
distintas. Los puntos deben estar lo suficientemente separados para una
alineación precisa. Puedes arrastrar cualquiera de los puntos después de
colocarlos para ajustar la posición.

Usa la rueda del ratón para acercar la vista previa y clic central con
arrastre para desplazarte. El botón **Restablecer** en la parte inferior
borra ambos puntos y te permite empezar de nuevo.

Una vez colocados ambos puntos, haz clic en **Siguiente** para continuar.

## Paso 2: Registrar posiciones físicas

![Registrar posiciones físicas](/screenshots/addon-print-and-cut-jog.png)

En esta página desplazas el láser a las posiciones físicas que corresponden a
los dos puntos de diseño que seleccionaste. El panel derecho muestra un pad
direccional para el desplazamiento y un control de distancia que establece
cuánto se mueve el láser por paso.

Desplaza el láser a la posición física que corresponde a tu primer punto de
diseño, luego haz clic en **Registrar** junto a Posición 1. Las coordenadas
registradas aparecen en la fila. Repite el proceso para Posición 2. Puedes
volver a visitar una posición registrada en cualquier momento haciendo clic
en el botón **Ir a** junto a ella.

El interruptor **Enfocar láser** enciende el láser con la potencia de enfoque
configurada, lo que crea un punto visible en el material para ayudarte a
localizar posiciones con precisión. Este interruptor requiere un valor de
potencia de enfoque mayor que cero en la configuración del láser.

La posición actual del láser se muestra en la parte inferior del panel.
Cuando ambas posiciones están registradas, haz clic en **Siguiente** para
continuar.

## Paso 3: Revisar y aplicar la transformación

![Revisar y aplicar la transformación](/screenshots/addon-print-and-cut-apply.png)

La última página muestra la alineación calculada como un desplazamiento de
traslación y un ángulo de rotación. Estos valores se derivan de la diferencia
entre tus puntos de diseño y las posiciones físicas registradas.

Por defecto, la escala está bloqueada en 1.0. Si tu material físico difiere
en tamaño del diseño — por ejemplo, debido a la escala de la impresora —
activa el interruptor **Permitir escalado**. El factor de escala se calcula
entonces a partir de la proporción entre la distancia física y la distancia de
diseño entre tus dos puntos. Aparece una nota cuando la escala está bloqueada
pero las distancias no coinciden, indicando que el segundo punto puede no
alinearse exactamente.

Haz clic en **Aplicar** para mover y rotar el workpiece en el lienzo para que
coincida con las posiciones físicas. La transformación se aplica como una
acción deshacible.

## Temas relacionados

- [Posicionamiento de workpieces](../features/workpiece-positioning) - Posicionar y transformar workpieces manualmente
- [Configuración del láser](../machine/laser) - Configurar la potencia de enfoque del láser
