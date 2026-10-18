---
description:
  "Un contador de trabajos en plantillas de texto: un número del proyecto que avanza tras cada trabajo,
  con números consecutivos para las copias de una matriz."
---

# Texto variable

Los cuadros de texto resuelven plantillas `{...}` (consulta
[Parámetros y expresiones](sketcher/expressions.md#template-expressions-in-text-boxes)). Además de
los parámetros del boceto y las funciones integradas, el proyecto ofrece variables que cambian de un
trabajo a otro, para que puedas grabar placas, etiquetas y rótulos numerados sin editar el texto
cada vez.

## Números de serie

Usa `{counter}` en un cuadro de texto. Es un número entero, así que los especificadores de formato
controlan su aspecto:

- `{counter}` — `7`
- `{counter:04d}` — `0007`, cuatro dígitos con ceros a la izquierda
- `No. {counter:03d}` — `No. 007`

Abre **Herramientas → Texto variable…** para configurar el contador:

- **Valor actual** — el número con el que empieza el próximo trabajo.
- **Valor inicial** — adonde vuelve **Restablecer**.
- **Avanzar en** — el incremento por copia y por trabajo.
- **Volver al llegar al valor final** / **Valor final** — después del valor final, el contador
  continúa en el valor inicial.
- **Avanzar tras cada trabajo** — cuando termina un trabajo enviado a la máquina, el contador
  avanza. El encuadre, las pasadas de prueba con el puntero y los trabajos cancelados no lo
  modifican.
- **Avanzar tras exportar G-code** — avanzar también al exportar G-code, para trabajos que se
  ejecutan desde una tarjeta de memoria.

El grupo **Próximo trabajo** muestra una vista previa de los números que usará el próximo trabajo.
**Anterior** y **Siguiente** mueven el contador un trabajo, **Restablecer** vuelve al valor inicial.
Cada cambio, incluido el avance automático, forma parte del historial de deshacer y se guarda en el
proyecto.

## Numerar copias

Cada pieza tiene un **Desplazamiento de variable** (en el panel de propiedades, en **Texto
variable**). Una copia con desplazamiento _n_ muestra el número _n_ copias después del actual. Una
[matriz en cuadrícula o circular](arrays.md) da a sus copias desplazamientos crecientes de forma
automática, así que una matriz de 20 placas graba 20 números consecutivos y el siguiente trabajo
continúa con el 21.

Las matrices creadas dentro de un boceto son distintas: sus copias pertenecen a una sola pieza y
todas muestran el mismo número. Crea la matriz en el lienzo para numerar las copias.

Para copias que hayas colocado tú, selecciónalas todas y elige **Organizar → Matriz → Numerar
copias**: los desplazamientos se asignan en orden de lectura, filas de arriba abajo y cada fila de
izquierda a derecha.

El contador avanza según el número de copias del trabajo: el mayor desplazamiento de variable más
uno.
