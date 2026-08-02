# Frente de Onda

El despeje adaptativo por frente de onda rellena formas vectoriales
cerradas con trayectorias de herramienta concéntricas que se expanden
hacia afuera desde el centro del hueco como ondas en un estanque. Los
anillos en expansión manejan automáticamente las islas interiores y
producen trayectorias suaves y continuas sin las inversiones bruscas
del escaneo por raster.

## Resumen

A diferencia del grabado raster tradicional, que barre de un lado a otro
en líneas paralelas, el frente de onda genera pasadas concéntricas que
irradian desde el centro de cada hueco. Esto produce un acabado uniforme
similar a ondas, adecuado para aplicaciones donde el patrón de relleno
contribuye al resultado visual.

Las operaciones de frente de onda:

- Rellenan formas vectoriales cerradas (huecos) con pasadas concéntricas
- Se expanden hacia afuera desde el centro del hueco
- Navegan automáticamente alrededor de islas interiores (agujeros dentro
  del hueco)
- Producen trayectorias suaves sin inversiones de dirección

## Cuándo Usar Frente de Onda

El frente de onda es un patrón de relleno alternativo para áreas de
huecos. Sus anillos concéntricos pueden ser visualmente más atractivos
que las líneas raster paralelas, y el patrón en expansión complementa
naturalmente las formas circulares u orgánicas.

Usa el despeje adaptativo por frente de onda para:

- Rellenar huecos en diseños vectoriales
- Fabricación de sellos y troqueles — el frente de onda despeja el hueco
  de fondo conservando las características en relieve como islas interiores
- Aplicaciones donde la textura de relleno es visible en la pieza terminada

**No uses frente de onda para:**

- Cortar a lo largo de contornos (usa [Contorno](contour) en su lugar)
- Rellenar imágenes de mapa de bits (usa [Grabado](engrave) en su lugar)
- Secciones de pared delgada donde no existe un hueco

## Crear una Operación de Frente de Onda

### Paso 1: Seleccionar Objetos

1. Importa o dibuja formas vectoriales cerradas en el lienzo
2. Selecciona los objetos que definen el límite del hueco
3. Asegúrate de que las formas sean trayectorias cerradas

### Paso 2: Añadir Operación de Frente de Onda

- **Menú:** Operaciones → Añadir Frente de Onda
- **Clic derecho:** Menú contextual → Añadir Operación → Frente de Onda

### Paso 3: Configurar Ajustes

Ajusta el paso y el desplazamiento según tu material y el acabado deseado.

![Resultado de operación de frente de onda](/screenshots/operations-wavefront.png)

## Ajustes Clave

El diálogo de ajustes de paso tiene tres pestañas: **Ajustes de Paso**, **Láser** y **Post-Procesamiento**. Los ajustes se describen en orden de pestañas a continuación.

### Ajustes de Frente de Onda

![Ajustes de paso de frente de onda](/screenshots/step-settings-wavefront-general.png)

El grupo **Frente de Onda** en la pestaña _Ajustes de Paso_ controla el patrón de relleno.

#### Paso

La distancia entre pasadas consecutivas de frente de onda (mm). Valores
más pequeños dan una cobertura más densa con más pasadas y tiempos de
trabajo más largos. Valores más grandes espacian más las pasadas para
una finalización más rápida.

**El Paso predeterminado es el tamaño del punto láser** y tiene un rango
de 0.05–50.0 mm.

| Paso    | Densidad de línea      | Tiempo de trabajo |
| ------- | ---------------------- | ----------------- |
| 0.1 mm  | Densa, muchas líneas   | Más lento         |
| 0.3 mm  | Moderada               | Medio             |
| 1.0 mm+ | Dispersa, menos líneas | Rápido            |

Los valores típicos son de 0.1–0.5 mm para la mayoría de las aplicaciones.

#### Desplazamiento

Espacio adicional desde la pared del hueco (mm). Crea un margen entre la
pasada de frente de onda más externa y el contorno del límite. Esto es
útil cuando una pasada de [Contorno](contour) separada terminará el borde,
o cuando se quiere dejar un borde deliberado alrededor del hueco.

Rango: 0.0–20.0 mm. El valor predeterminado es 0.0 (las pasadas de frente
de onda se extienden hasta el límite).

### Ajustes del Láser

![Ajustes del láser](/screenshots/step-settings-wavefront-laser.png)

La potencia, la velocidad y la selección del cabezal láser se encuentran
en la página **Láser** del diálogo de ajustes de paso.

**Potencia (%):**

- Intensidad del láser para cortar
- Coincide con los requisitos de corte de tu material

**Velocidad (mm/min):**

- Qué tan rápido se mueve el láser
- Coincide con la velocidad de corte de tu material

## Cómo Funciona el Frente de Onda

1. **Pasada de entrada** — Una entrada helicoidal se introduce en el
   centro del hueco para establecer un área despejada inicial
2. **Expansión del frente de onda** — Comenzando desde el centro
   despejado, los anillos concéntricos se expanden hacia afuera. Cada
   anillo se extiende más allá del anterior en la distancia de paso
   configurada
3. **Manejo de islas** — A medida que el frente de onda crece, encuentra
   y rodea las islas interiores, dejándolas en pie
4. **Finalización** — La expansión continúa hasta que toda el área del
   hueco está cubierta

## Post-Procesamiento

![Ajustes de post-procesamiento de frente de onda](/screenshots/step-settings-wavefront-post.png)

Las operaciones de frente de onda soportan:

- **[Suavizado de Trayectoria](../smooth.md)** — Reduce los bordes dentados
  en las trayectorias de herramienta
- **[Optimización de Trayectoria](../path-optimization.md)** — Minimiza la
  distancia de desplazamiento entre pasadas

## Consejos y Mejores Prácticas

### Elegir el Paso

- Una cobertura más densa (paso pequeño) significa más pasadas y tiempos
  de trabajo más largos
- Una cobertura dispersa (paso grande) es más rápida pero deja más
  material entre pasadas
- Equilibra la densidad con el tiempo de trabajo según tu aplicación

### Fabricación de Sellos y Troqueles

El frente de onda es muy adecuado para la fabricación de sellos. Los
anillos concéntricos en expansión despejan naturalmente el hueco de
fondo mientras navegan alrededor de las características en relieve
tratadas como islas interiores.

### Combinación con Contorno

Un flujo de trabajo común es despejar el interior del hueco con frente
de onda y luego terminar el límite con una pasada de [Contorno](contour)
para un borde limpio. Ajusta el desplazamiento para dejar suficiente
margen para el corte de contorno.

## Temas Relacionados

- **[Contorno](contour)** — Corte a lo largo de contornos vectoriales
- **[Grabado](engrave)** — Relleno de áreas con patrones de grabado raster
- **[Envoltura Ajustada](shrink-wrap)** — Corte de límite alrededor de
  objetos
- **[Suavizado de Trayectoria](../smooth.md)** — Refinamiento de bordes de
  trayectoria de herramienta
