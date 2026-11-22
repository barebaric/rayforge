# Relleno por Desplazamiento

El relleno por desplazamiento rellena formas vectoriales cerradas con anillos paralelos a su
contorno. El primer anillo sigue el propio contorno; cada anillo siguiente queda un intervalo de
línea más hacia dentro, hasta que la forma queda rellena. Los agujeros de una forma se mantienen
libres: los anillos también siguen sus bordes.

## Resumen

Mientras que el [Grabado](engrave) barre de un lado a otro en líneas rectas y el
[Frente de Onda](wavefront) se expande desde el centro del hueco, el relleno por desplazamiento
trabaja desde el contorno hacia dentro. Cada anillo es una trayectoria cerrada que se corta a lo
largo de su línea central, de modo que el relleno adopta la forma del diseño.

Las operaciones de relleno por desplazamiento:

- Rellenan formas vectoriales cerradas con anillos paralelos al contorno
- Mantienen libres los agujeros dentro de una forma
- Ignoran las trayectorias abiertas
- Cortan los anillos de fuera hacia dentro o de dentro hacia fuera

## Cuándo Usar Relleno por Desplazamiento

Usa el relleno por desplazamiento para:

- Rellenar texto, logotipos y otras formas vectoriales con un relleno que sigue sus contornos
- Vaciar huecos donde los anillos deben ajustarse al borde de la forma
- Diseños donde el patrón de relleno es visible en la pieza terminada

**No uses el relleno por desplazamiento para:**

- Cortar solo a lo largo de contornos (usa [Contorno](contour) en su lugar)
- Rellenar imágenes de mapa de bits (usa [Grabado](engrave) en su lugar)
- Trayectorias abiertas, que no encierran un área que rellenar

## Crear una Operación de Relleno por Desplazamiento

### Paso 1: Preparar las Formas

1. Importa o dibuja formas vectoriales cerradas en el lienzo
2. Asegúrate de que las formas sean trayectorias cerradas; las trayectorias abiertas se omiten

### Paso 2: Añadir un Paso de Relleno por Desplazamiento

1. Abre el flujo de trabajo de la capa en el panel derecho.
2. Haz clic en el botón **Añadir paso** y elige **Relleno por Desplazamiento**.

### Paso 3: Configurar Ajustes

Ajusta el intervalo de línea, el desplazamiento y la dirección de relleno según tu material y el
acabado deseado.

![Resultado de la operación de Relleno por Desplazamiento](/screenshots/operations-offset-fill.webp)

## Ajustes Clave

El diálogo de ajustes del paso tiene cuatro pestañas: **Ajustes de Paso**, **Láser**,
**Post-Procesamiento** y **Piezas de trabajo**. Los ajustes se describen en orden de pestañas a
continuación.

### Ajustes de Relleno por Desplazamiento

![Ajustes del paso de Relleno por Desplazamiento](/screenshots/step-settings-offset-fill-general.webp)

El grupo **Relleno por Desplazamiento** de la pestaña _Ajustes del paso_ controla el patrón de
relleno.

#### Intervalo de Línea (Line Interval)

La distancia entre anillos de relleno vecinos. Los valores más pequeños dan un relleno más denso con
más anillos y trabajos más largos; los valores más grandes son más rápidos pero dejan más espacio
entre los anillos.

**El intervalo de línea es por defecto el tamaño del punto láser** y tiene un rango de 0,01–50,0 mm.

#### Desplazamiento (Offset)

La distancia del primer anillo dentro del contorno. Con el valor por defecto de 0,0 el primer anillo
queda sobre el contorno; un valor mayor deja un margen entre el contorno y el relleno. Es útil
cuando una pasada de [Contorno](contour) separada terminará el borde.

Rango: 0,0–20,0 mm. El valor por defecto es 0,0.

#### Dirección de Relleno (Fill Direction)

El orden en que se cortan los anillos:

- **Exterior-Interior** (por defecto): el anillo más exterior se corta primero y el relleno avanza
  hacia el centro
- **Interior-Exterior**: el anillo más interior se corta primero y el anillo del contorno al final

### Ajustes del Láser

![Ajustes del Láser](/screenshots/step-settings-offset-fill-laser.webp)

La potencia, la velocidad y la selección del cabezal láser están en la página **Láser** del diálogo
de ajustes del paso. Un nuevo paso de relleno por desplazamiento empieza con un 80 % de potencia.

**Potencia (%):**

- Intensidad del láser para el relleno
- Ajústala a los requisitos de tu material

**Velocidad (mm/min):**

- Qué tan rápido se mueve el láser
- Ajústala a la velocidad de grabado de tu material

## Cómo Funciona el Relleno por Desplazamiento

1. **Formas cerradas** — Las trayectorias abiertas se descartan; cada forma cerrada se trata junto
   con sus agujeros
2. **Primer anillo** — El contorno, desplazado hacia dentro según el desplazamiento configurado
3. **Anillos siguientes** — Cada anillo siguiente queda un intervalo de línea dentro del anterior;
   los anillos alrededor de los agujeros crecen a la vez hacia fuera desde los agujeros
4. **Finalización** — Se añaden anillos hasta que la forma queda rellena
5. **Corte** — Los anillos se cortan a lo largo de su línea central en la dirección de relleno
   elegida

## Post-Procesamiento

![Ajustes de post-procesamiento de Relleno por Desplazamiento](/screenshots/step-settings-offset-fill-post.webp)

Las operaciones de relleno por desplazamiento admiten:

- **[Suavizado de Trayectoria](../smooth.md)** — Reduce los bordes dentados en las trayectorias
  (desactivado por defecto)
- **[Recortar al Material](../crop-to-stock.md)** — Limita las trayectorias al límite del material
  (desactivado por defecto)
- **[Optimización de Trayectoria](../path-optimization.md)** — Minimiza la distancia de
  desplazamiento entre anillos
- **[Multi-Pasada](../multi-pass.md)** — Repite el relleno varias veces

## Consejos y Mejores Prácticas

### Elegir el Intervalo de Línea

- Empieza con el valor por defecto (el tamaño del punto láser) para un relleno cerrado y uniforme
- Aumenta el intervalo para un patrón de anillos visible o un trabajo más rápido

### Combinación con Contorno

Vacía el interior de una forma con el relleno por desplazamiento y termina luego el borde con una
pasada de [Contorno](contour). Ajusta el desplazamiento para que el relleno deje espacio suficiente
para el corte de contorno.

## Temas Relacionados

- **[Frente de Onda](wavefront)** — Relleno concéntrico que se expande desde el centro del hueco
- **[Contorno](contour)** — Corte a lo largo de contornos vectoriales
- **[Grabado](engrave)** — Relleno de áreas con patrones de grabado raster
- **[Suavizado de Trayectoria](../smooth.md)** — Refinamiento de bordes de trayectoria
