# Sistemas de Coordenadas de Trabajo (WCS)

Los Sistemas de Coordenadas de Trabajo (WCS) te permiten definir múltiples puntos de referencia en el área de trabajo de tu máquina. Esto facilita ejecutar el mismo trabajo en diferentes posiciones sin rediseñar o reposicionar tus piezas.

## Espacios de Coordenadas

Rayforge utiliza tres espacios de coordenadas que trabajan juntos:

| Espacio      | Descripción                                                                                                            |
| ------------ | ---------------------------------------------------------------------------------------------------------------------- |
| **MACHINE**  | Coordenadas absolutas relativas a la posición de origen de la máquina. El origen está fijado por hardware.             |
| **WORKAREA** | El área utilizable dentro de tu máquina, considerando los márgenes alrededor de la cama.                               |
| **WCS**      | El sistema de coordenadas de tu trabajo. Origen configurable por el usuario para diseño y posicionamiento de trabajos. |

:::note Nota para Desarrolladores
Internamente, Rayforge utiliza un sistema de coordenadas normalizado llamado espacio WORLD.
El espacio WORLD describe el mismo espacio físico que el espacio MACHINE, pero con una
convención fija: Y hacia arriba con origen en la esquina inferior izquierda. Esto simplifica
los cálculos internos y el renderizado. Los usuarios no necesitan interactuar directamente
con el espacio WORLD.
:::

### Espacio MACHINE

El espacio MACHINE es el sistema de coordenadas absoluto relativo a la posición de origen
de tu máquina. El origen (0,0) está determinado por la configuración de homing de tu máquina.

- **Origen**: Posición de origen de la máquina (0,0,0) - fijado por hardware
- **Propósito**: Referencia para todos los demás sistemas de coordenadas
- **Fijo**: No puede ser cambiado por software

La dirección de las coordenadas depende de la configuración de tu máquina:

- **Esquina de origen**: Puede ser superior-izquierda, inferior-izquierda, superior-derecha o inferior-derecha
- **Dirección de ejes**: Los ejes X e Y pueden invertirse según la configuración del hardware

### Espacio WORKAREA

El espacio WORKAREA define el área utilizable dentro de tu máquina, considerando
cualquier margen alrededor de los bordes de tu cama.

- **Origen**: Misma esquina que el origen del espacio MACHINE
- **Propósito**: Define el área real donde pueden ejecutarse los trabajos
- **Márgenes**: Pueden aplicarse márgenes (izquierda, superior, derecha, inferior)

Por ejemplo, si tu máquina es de 400×300mm pero tiene un margen de 10mm en todos los lados,
la WORKAREA sería de 380×280mm comenzando en la posición (10, 10) en el espacio MACHINE.

## Entendiendo WCS

Piensa en WCS como "puntos cero" personalizables para tu trabajo. Mientras que tu máquina tiene una posición de origen fija (determinada por los interruptores de límite), WCS te permite definir dónde quieres que comience tu trabajo.

### ¿Por Qué Usar WCS?

- **Múltiples fijaciones**: Configura varias áreas de trabajo en tu cama y cambia entre ellas
- **Posicionamiento repetible**: Ejecuta el mismo trabajo en diferentes ubicaciones
- **Alineación rápida**: Establece un punto de referencia basado en tu material o pieza
- **Flujos de trabajo de producción**: Organiza múltiples trabajos a través de tu área de trabajo

## Tipos de WCS

Rayforge soporta los siguientes sistemas de coordenadas:

| Sistema | Tipo      | Descripción                                                    |
| ------- | --------- | -------------------------------------------------------------- |
| **G53** | Máquina   | Coordenadas absolutas de máquina (fijas, no se pueden cambiar) |
| **G54** | Trabajo 1 | Primer sistema de coordenadas de trabajo (por defecto)         |
| **G55** | Trabajo 2 | Segundo sistema de coordenadas de trabajo                      |
| **G56** | Trabajo 3 | Tercer sistema de coordenadas de trabajo                       |
| **G57** | Trabajo 4 | Cuarto sistema de coordenadas de trabajo                       |
| **G58** | Trabajo 5 | Quinto sistema de coordenadas de trabajo                       |
| **G59** | Trabajo 6 | Sexto sistema de coordenadas de trabajo                        |

### Coordenadas de Máquina (G53)

G53 representa la posición absoluta de tu máquina, con cero en la posición de origen de la máquina. Esto está fijado por tu hardware y no se puede cambiar.

**Cuándo usar:**

- Homing y calibración
- Posicionamiento absoluto relativo a límites de la máquina
- Cuando necesitas referenciar la posición física de la máquina

### Coordenadas de Trabajo (G54-G59)

Estos son sistemas de coordenadas desplazados que puedes definir. Cada uno tiene su propio punto cero que puedes establecer en cualquier lugar de tu área de trabajo.

**Cuándo usar:**

- Configurar múltiples fijaciones de trabajo
- Alinear a posiciones de material
- Ejecutar el mismo trabajo en diferentes ubicaciones

## Visualizando WCS en la Interfaz

### Lienzo 2D

El lienzo 2D muestra tu origen WCS con un marcador verde:

- **Líneas verdes**: Indican la posición del origen WCS actual (0, 0)
- **Alineación de cuadrícula**: Las líneas de la cuadrícula están alineadas al origen WCS, no al origen de máquina

El marcador de origen se mueve cuando cambias el WCS activo o su desplazamiento, mostrándote exactamente dónde comenzará tu trabajo.

### Vista Previa 3D

En la vista previa 3D, WCS se muestra diferente:

- **Cuadrícula y ejes**: Toda la cuadrícula aparece como si el origen WCS fuera el origen del mundo
- **Vista aislada**: El WCS se muestra "en aislamiento" - parece que la cuadrícula está centrada en el WCS, no en la máquina
- **Etiquetas**: Las etiquetas de coordenadas son relativas al origen WCS

Esto facilita visualizar dónde se ejecutará tu trabajo relativo al sistema de coordenadas de trabajo seleccionado.

## Seleccionando y Cambiando WCS

### Vía la Barra de Herramientas

1. Ubica el menú desplegable WCS en la barra de herramientas principal (etiquetado "G53" por defecto)
2. Haz clic para ver los sistemas de coordenadas disponibles
3. Selecciona el WCS que quieres usar

### Vía el Panel de Control

1. Abre el Panel de Control (Ver → Panel de Control o Ctrl+L)
2. Encuentra el menú desplegable WCS en la sección de estado de la máquina
3. Selecciona el WCS deseado del menú desplegable

## Estableciendo Desplazamientos WCS

Puedes definir dónde está ubicado el origen de cada WCS en tu máquina.

### Estableciendo Cero en la Posición Actual

1. Conéctate a tu máquina
2. Selecciona el WCS que quieres configurar (ej., G54)
3. Desplaza la cabeza del láser a la posición que quieres que sea (0, 0)
4. En el Panel de Control, haz clic en los botones de cero:
   - **Cero X**: Establece la posición X actual como 0 para el WCS activo
   - **Cero Y**: Establece la posición Y actual como 0 para el WCS activo
   - **Cero Z**: Establece la posición Z actual como 0 para el WCS activo

Los desplazamientos se almacenan en el controlador de tu máquina y persisten entre sesiones.

### Viendo Desplazamientos Actuales

El Panel de Control muestra los desplazamientos actuales para el WCS activo:

- **Desplazamientos Actuales**: Muestra el desplazamiento (X, Y, Z) desde el origen de la máquina
- **Posición Actual**: Muestra la posición de la cabeza del láser en el WCS activo

## WCS en tus Trabajos

Cuando ejecutas un trabajo, Rayforge usa el WCS activo para posicionar tu trabajo:

1. Diseña tu trabajo en el lienzo
2. Selecciona el WCS que quieres usar
3. Ejecuta el trabajo - será posicionado según el desplazamiento WCS

El mismo trabajo puede ejecutarse en diferentes posiciones simplemente cambiando el WCS activo.

## Flujos de Trabajo Prácticos

### Flujo de Trabajo 1: Múltiples Posiciones de Fijación

Tienes una cama grande y quieres configurar tres áreas de trabajo:

1. **Haz homing a tu máquina** para establecer una referencia
2. **Desplázate a la primera área de trabajo** y establece el desplazamiento G54 (Cero X, Cero Y)
3. **Desplázate a la segunda área de trabajo** y establece el desplazamiento G55
4. **Desplázate a la tercera área de trabajo** y establece el desplazamiento G56
5. Ahora puedes cambiar entre G54, G55 y G56 para ejecutar trabajos en cada área

### Flujo de Trabajo 2: Alineando a Material

Tienes una pieza de material colocada en algún lugar de tu cama:

1. **Desplaza la cabeza del láser** a la esquina de tu material
2. **Selecciona G54** (o tu WCS preferido)
3. **Haz clic en Cero X y Cero Y** para establecer la esquina del material como (0, 0)
4. **Diseña tu trabajo** con (0, 0) como origen
5. **Ejecuta el trabajo** - comenzará desde la esquina del material

### Flujo de Trabajo 3: Cuadrícula de Producción

Necesitas cortar la misma pieza 10 veces en diferentes ubicaciones:

1. **Diseña una pieza** en Rayforge
2. **Configura los desplazamientos G54-G59** para tus posiciones deseadas
3. **Ejecuta el trabajo** con G54 activo
4. **Cambia a G55** y ejecuta de nuevo
5. **Repite** para cada posición WCS

## Notas Importantes

### Limitaciones de WCS

- **G53 no puede cambiarse**: Las coordenadas de máquina están fijadas por hardware
- **Los desplazamientos persisten**: Los desplazamientos WCS se almacenan en el controlador de tu máquina
- **Conexión requerida**: Debes estar conectado a una máquina para establecer desplazamientos WCS

---

**Páginas Relacionadas:**

- [Panel de Control](../ui/control-panel) - Control manual y gestión de WCS
- [Configuración de Máquina](../machine/general) - Configura tu máquina
- [Vista Previa 3D](../ui/3d-preview) - Visualizando tus trabajos
