---
description:
  "El paso de comando inyecta código de máquina personalizado, una línea por comando, en una
  posición exacta del flujo de trabajo de una capa. Úsalo para preposicionamiento, control de
  accesorios y comandos específicos de la máquina."
---

# Comando

El paso de comando inyecta código de máquina personalizado en el trabajo, exactamente en la posición
que ocupa el paso en el flujo de trabajo de la capa. Úsalo para enviar comandos que las operaciones
de geometría no cubren: posicionar el cabezal, activar o desactivar la asistencia de aire u otro
equipo alrededor de operaciones específicas, o enviar códigos específicos de la máquina.

![Configuración del paso de comando](/screenshots/step-settings-command-general.webp)

## Descripción general

El paso de comando:

- Contiene un bloque de código de máquina de varias líneas, un comando por línea
- Emite cada línea textualmente en la posición del paso cuando se codifica el trabajo
- Se ejecuta una vez por capa, en su posición del flujo de trabajo — no actúa sobre las piezas
- Funciona sin ninguna pieza en la capa, de modo que una capa que contenga solo un paso de comando
  aún puede generar un trabajo
- Admite las mismas variables de ruta que las macros (ver más abajo)

El texto se guarda sin expandir en el proyecto, por lo que viaja con el archivo `.ryp` y documenta
exactamente qué se envía y a dónde.

## Cuándo usar el paso de comando

Usa el paso de comando para:

- Preposicionar el cabezal (p. ej., elevar Z antes de que comience un corte)
- Activar o desactivar la asistencia de aire, el refrigerante u otro equipo entre operaciones
- Enviar códigos específicos de la controladora alrededor de un trabajo
- Pausar brevemente con una dwell entre dos operaciones

**No uses el paso de comando para:**

- Repetir código G en los límites de capa o de pieza — las
  [Macros y Hooks](../../machine/hooks-macros.md) se activan automáticamente y también funcionan en
  máquinas sin código G
- Ejecutar programas en la computadora (eso no es compatible en esta fase)

## Agregar un paso de comando

1. Abre el flujo de trabajo de la capa en el panel derecho.
2. Haz clic en el botón **Añadir paso** y elige **Comando**.
3. Ingresa el código de máquina en el cuadro de texto, un comando por línea. Las líneas vacías se
   omiten.

El paso puede colocarse antes, entre o después de otros pasos, y puede usarse varias veces en una
capa. Su posición en el flujo de trabajo es la posición que ocupan sus líneas en el código de
máquina generado.

## Variables de ruta

Como las macros, las líneas pueden contener variables de ruta que se resuelven cuando se codifica el
trabajo. Las variables desconocidas se dejan intactas, y las variables `layer.*` se resuelven
incluso cuando el paso se ejecuta a mitad de una capa.

| Variable           | Ejemplo    | Descripción                                                   |
| ------------------ | ---------- | ------------------------------------------------------------- |
| `{machine.name}`   | `My Laser` | Nombre de la máquina activa                                   |
| `{layer.name}`     | `Layer 1`  | Nombre de la capa que se está procesando                      |
| `{job.extents[0]}` | `210.0`    | Extensión del trabajo en el eje X (mm)                        |
| `{wcs_offset[0]}`  | `5.0`      | Desplazamiento X del sistema de coordenadas de trabajo activo |

Por ejemplo, un comentario como `; cortando {layer.name} en {machine.name}` se codifica con los
nombres sustituidos.

## Compatibilidad de máquinas

El paso de comando está disponible en todas las máquinas, pero las líneas solo se emiten para los
drivers que consumen código de máquina. Cuando el driver de la máquina activa no lo hace (p. ej.,
Ruida), el paso muestra una advertencia y sus líneas no forman parte de la salida generada.
