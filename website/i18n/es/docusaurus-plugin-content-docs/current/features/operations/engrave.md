# Grabado

Las operaciones de grabado rellenan áreas con líneas de escaneo rasterizado, soportando múltiples
modos para diferentes efectos de grabado. Desde fotos suaves en escala de grises hasta efectos de
relieve 3D, elige el modo que mejor se adapte a tu diseño y material.

## Resumen

Las operaciones de grabado:

- Rellenan formas cerradas con líneas de escaneo
- Soportan múltiples modos de grabado para diferentes efectos
- Funcionan tanto con formas vectoriales como con imágenes de mapa de bits
- Usan escaneo bidireccional para velocidad
- Crean marcas permanentes en muchos materiales

## Modos de Grabado

### Modo de Potencia Variable

El modo de Potencia Variable varía la potencia del láser continuamente basándose en el brillo de la
imagen, creando un grabado suave en escala de grises con transiciones graduales.

**Mejor Para:**

- Fotos e imágenes suaves en escala de grises
- Degradados y transiciones naturales
- Retratos y obras de arte
- Grabado en madera y cuero

**Características Clave:**

- Modulación continua de potencia
- Control de potencia mín/máx
- Degradados suaves
- Mejor calidad tonal que el tramado

### Modo de Potencia Constante

El modo de Potencia Constante graba a plena potencia, con un umbral que determina qué píxeles se
graban. Esto crea resultados limpios en blanco/negro.

**Mejor Para:**

- Texto y logos
- Gráficos de alto contraste
- Grabados limpios en blanco/negro
- Formas y patrones simples

**Características Clave:**

- Grabado basado en umbral
- Salida de potencia consistente
- Más rápido que el modo de potencia variable
- Bordes limpios

### Modo Trama

El modo Trama convierte imágenes en escala de grises a patrones binarios usando algoritmos de
tramado, permitiendo un grabado de fotos de alta calidad con mejor reproducción tonal que los
métodos simples basados en umbral.

**Mejor Para:**

- Grabar fotografías en madera o cuero
- Crear obras de arte estilo media tinta
- Imágenes con degradados suaves
- Cuando el rasterizado estándar no captura suficiente detalle

**Características Clave:**

- Múltiples opciones de algoritmos de tramado
- Mejor preservación del detalle
- Tonos continuos percibidos
- Ideal para fotografías

### Modo Múltiples Profundidades

El modo Múltiples Profundidades crea efectos de relieve 3D variando la potencia del láser basándose
en el brillo de la imagen, con múltiples pasadas para un tallado más profundo.

**Mejor Para:**

- Crear retratos y obras de arte 3D
- Mapas de terreno y topográficos
- Litofanías (imágenes 3D que transmiten luz)
- Logos y diseños en relieve
- Esculturas en relieve

**Características Clave:**

- Mapeo de profundidad desde el brillo de la imagen
- Profundidad mín/máx configurable
- Degradados suaves
- Múltiples pasadas para un grabado más profundo
- Escalonamiento Z entre pasadas

## Cuándo Usar Grabado

Usa operaciones de grabado para:

- Grabar texto y logos
- Crear imágenes y fotos en madera/cuero
- Rellenar áreas sólidas con textura
- Marcar piezas y productos
- Crear efectos de relieve 3D
- Obras de arte estilo media tinta

**No uses grabado para:**

- Cortar a través del material (usa [Contorno](contour) en su lugar)
- Contornos precisos (el rasterizado crea áreas rellenas)
- Trabajo de líneas finas (los vectores son más limpios)

## Crear una Operación de Grabado

### Paso 1: Preparar el Contenido

El grabado funciona con:

- **Formas vectoriales** - Rellenas con líneas de escaneo
- **Texto** - Convertido a trayectorias rellenas
- **Imágenes** - Convertidas a escala de grises y grabadas

### Paso 2: Añadir Operación de Grabado

- **Menú:** Operaciones → Añadir Grabado
- **Atajo:** <kbd>ctrl+shift+e</kbd>
- **Clic derecho:** Menú contextual → Añadir Operación → Grabado

### Paso 3: Elegir Modo

Selecciona el modo de grabado que mejor se adapte a tus necesidades:

- **Potencia Variable** - Grabado suave en escala de grises
- **Potencia Constante** - Grabado limpio en blanco/negro
- **Trama** - Grabado de fotos de alta calidad
- **Múltiples Profundidades** - Efectos de relieve 3D

### Paso 4: Configurar Ajustes

![Ajustes de paso de grabado](/screenshots/step-settings-engrave-general-variable.webp)

## Ajustes de Grabado

Los grupos **Grabado** y **Potencia** en la pestaña _Ajustes de Paso_ controlan el patrón de
escaneo, el modo y la modulación de potencia, en orden de filas. La potencia y la velocidad del
láser se encuentran en la página **Láser** (ver abajo).

### Modo

La fila **Modo** selecciona uno de los cuatro modos de grabado. Cada modo expone diferentes ajustes,
descritos a continuación.

### Ajustes Específicos del Modo

#### Ajustes del Modo Potencia Variable

![Ajustes del modo Potencia Variable](/screenshots/step-settings-engrave-general-variable.webp)

**Potencia Mín (%):**

- Potencia del láser para las áreas más claras (píxeles blancos)
- Usualmente 0-20%
- Configura más alto para evitar áreas muy superficiales

**Potencia Máx (%):**

- Potencia del láser para las áreas más oscuras (píxeles negros)
- Usualmente 40-80% dependiendo del material
- Menor = relieve sutil, mayor = profundidad dramática

**Ejemplos de Rango de Potencia:**

| Mín | Máx | Efecto                      |
| --- | --- | --------------------------- |
| 0%  | 40% | Relieve sutil, ligero       |
| 10% | 60% | Profundidad media, seguro   |
| 20% | 80% | Profundo, relieve dramático |

**Invertir:**

- **Apagado** (predeterminado): Blanco = superficial, Negro = profundo
- **Encendido**: Blanco = profundo, Negro = superficial

Usa invertir para litofanías (las áreas claras deben ser delgadas) o repujado (áreas elevadas).

**Rango de Brillo:**

Controla cómo se mapean los valores de brillo de la imagen a la potencia del láser. El histograma
muestra la distribución de valores de brillo en tu imagen.

- **Auto Niveles** (predeterminado): Ajusta automáticamente los puntos de negro y blanco basándose
  en el contenido de la imagen. Los valores por debajo del punto de negro se tratan como negro, los
  valores por encima del punto de blanco se tratan como blanco. Esto estira el contraste de la
  imagen para usar todo el rango de potencia.
- **Modo Manual**: Deshabilita Auto Niveles para configurar manualmente los puntos de negro y blanco
  arrastrando los marcadores en el histograma.

Esto es particularmente útil para:

- Imágenes de bajo contraste que necesitan mejora de contraste
- Imágenes con rango tonal limitado
- Asegurar resultados consistentes entre diferentes imágenes fuente

#### Ajustes del Modo Potencia Constante

![Ajustes del modo Potencia Constante](/screenshots/step-settings-engrave-general-constant_power.webp)

**Umbral (0-255):**

- Corte de brillo para la separación blanco/negro
- Menor = más negro grabado
- Mayor = más blanco grabado

**Valores típicos:**

- 128 (umbral de 50% de gris)
- Ajustar basándose en el contraste de la imagen

#### Ajustes del Modo Trama

![Ajustes del modo Trama](/screenshots/step-settings-engrave-general-dither.webp)

**Algoritmo de Tramado:**

Elige el algoritmo que mejor se adapte a tu imagen y material:

| Algoritmo       | Calidad | Velocidad  | Mejor Para                         |
| --------------- | ------- | ---------- | ---------------------------------- |
| Floyd-Steinberg | Máxima  | Más lento  | Fotos, retratos, degradados suaves |
| Bayer 2x2       | Baja    | Más rápido | Efecto de media tinta grueso       |
| Bayer 4x4       | Media   | Rápido     | Media tinta equilibrada            |
| Bayer 8x8       | Alta    | Media      | Detalle fino, patrones sutiles     |

**Floyd-Steinberg** es el predeterminado y el recomendado para la mayoría de los grabados de fotos.
Usa difusión de error para distribuir los errores de cuantización a los píxeles vecinos, creando
resultados de aspecto natural.

**El tramado Bayer** crea patrones regulares que pueden producir efectos artísticos que asemejan la
impresión tradicional de media tinta.

#### Ajustes del Modo Múltiples Profundidades

![Ajustes del modo Múltiples Profundidades](/screenshots/step-settings-engrave-general-multi_pass.webp)

**Número de Niveles de Profundidad:**

- Número de niveles de profundidad discretos
- Más niveles = degradados más suaves
- Típico: 5-10 niveles

**Paso Z por Nivel (mm):**

- Cuánto bajar entre pasadas de profundidad
- Crea una profundidad total mayor con múltiples pasadas
- Típico: 0.1-0.5mm

**Rotar Ángulo por Pasada:**

- Grados para rotar cada pasada sucesiva
- Crea un efecto 3D tipo entramado cruzado
- Típico: 0-45 grados

**Invertir:**

- **Habilitado:** Blanco = profundo, Negro = superficial
- **Deshabilitado:** Negro = profundo, Blanco = superficial

Usa invertir para litofanías (las áreas claras deben ser delgadas) o repujado (áreas elevadas).

### Patrón de Escaneo

#### Intervalo de Línea

**Intervalo de Línea (mm):**

- Espaciado entre líneas de escaneo
- Menor = mayor calidad, tiempo de trabajo más largo
- Mayor = más rápido, líneas visibles

| Intervalo | Calidad | Velocidad  | Usar Para                  |
| --------- | ------- | ---------- | -------------------------- |
| 0.05mm    | Máxima  | Más lento  | Fotos, detalle fino        |
| 0.1mm     | Alta    | Media      | Texto, logos, gráficos     |
| 0.2mm     | Media   | Rápido     | Rellenos sólidos, texturas |
| 0.3mm+    | Baja    | Más rápido | Borrador, pruebas          |

**Recomendado:** 0.1mm para uso general

<!-- prettier-ignore-start -->
:::tip[Coincidencia de Resolución]
Para imágenes, el intervalo de línea debe coincidir o exceder la resolución de la imagen. Si tu
imagen es de 10 píxeles/mm (254 DPI), usa un intervalo de línea de 0.1mm o menor.
:::
<!-- prettier-ignore-end -->

#### Dirección de Escaneo

**Ángulo de Escaneo (grados):**

- Dirección de las líneas de escaneo
- 0 = horizontal (izquierda a derecha)
- 90 = vertical (arriba a abajo)
- 45 = diagonal

**¿Por qué cambiar el ángulo?**

- Veta de la madera: Graba perpendicular a la veta para mejores resultados
- Orientación del patrón: Coincidir con la estética del diseño
- Reducir bandas: Un ángulo diferente puede ocultar imperfecciones

**Escaneo Bidireccional:**

Rayforge siempre escanea de forma bidireccional (de izquierda a derecha y luego de derecha a
izquierda), ya que disparar en cada pasada aproximadamente duplica la velocidad de grabado comparado
con regresar sin disparar entre líneas.

Pequeñas diferencias mecánicas o de retardo de disparo entre las dos direcciones pueden causar
bandas visibles en algunas máquinas. Si ves esto, calibra el **Desplazamiento de Escaneo
Bidireccional** de abajo para corregirlo directamente, en lugar de perder el beneficio de velocidad.

#### Desplazamiento de Escaneo Bidireccional

Corrige un sesgo fijo mecánico o de retardo de disparo entre las pasadas raster de izquierda a
derecha y de derecha a izquierda, que de otro modo desalinea filas de escaneo alternas (visible como
bandas, especialmente en grabados de fotos).

- Se configura en milímetros, positivo o negativo dependiendo de qué dirección necesita desplazarse
- Aplica un desplazamiento constante independientemente de la velocidad; si el sesgo varía con la
  velocidad, calibra para tu velocidad de grabado típica
- El valor predeterminado es 0 (sin corrección)

**Calibrando el desplazamiento:**

1. Graba un patrón de prueba con detalle vertical visible (por ejemplo, una cuadrícula fina) usando
   escaneo bidireccional
2. Compara las filas alternas para encontrar la dirección y la cantidad de desalineación
3. Ajusta el desplazamiento en pequeños incrementos (0.01-0.05mm) y vuelve a probar hasta que las
   filas alternas se alineen

## Ajustes del Láser

![Ajustes del láser](/screenshots/step-settings-engrave-laser.webp)

La potencia, la velocidad y la selección del cabezal láser se encuentran en la página **Láser** del
diálogo de ajustes de paso.

### Potencia y Velocidad

**Potencia (%):**

- Intensidad del láser para grabar
- Menor potencia para un marcado más ligero
- Mayor potencia para un grabado más profundo

**Velocidad (mm/min):**

- Qué tan rápido escanea el láser
- Más rápido = más ligero, más lento = más oscuro

## Post-Procesamiento

![Ajustes de post-procesamiento de grabado](/screenshots/step-settings-engrave-post.webp)

Las operaciones de grabado soportan varias opciones de post-procesamiento:

- **[Overscan](../overscan.md)** - Extiende las líneas rasterizadas para una calidad de grabado
  consistente
- **[Optimización de Trayectoria](../path-optimization.md)** - Reduce la distancia de viaje
- **[Multi-Pasada](../multi-pass.md)** - Repite el grabado para resultados más profundos

### Overscan

**Distancia de Overscan (mm):**

- Qué tan más allá del diseño viaja el láser antes de dar la vuelta
- Permite que el láser alcance la velocidad completa antes de entrar al diseño
- Previene marcas de quemadura al inicio/final de las líneas

**Valores típicos:**

- 2-5mm para la mayoría de los trabajos
- Mayor para velocidades altas

Ver [Overscan](../overscan.md) para detalles.

## Consejos y Mejores Prácticas

### Selección de Material

**Mejores materiales para grabar:**

- Madera (las variaciones naturales crean resultados hermosos)
- Cuero (se quema a marrón oscuro/negro)
- Aluminio anodizado (remueve el recubrimiento, revela el metal)
- Metales recubiertos (remueve la capa de recubrimiento)
- Algunos plásticos (¡prueba primero!)

**Materiales desafiantes:**

- Acrílico transparente (no muestra bien el grabado)
- Metales sin recubrimiento (requieren compuestos de marcado especiales)
- Vidrio (requiere ajustes/recubrimientos especiales)

### Ajustes de Calidad

**Para mejor calidad:**

- Usa un intervalo de línea más pequeño (0.05-0.1mm)
- Calibra el Desplazamiento de Escaneo Bidireccional si ves bandas
- Aumenta el overscan (3-5mm)
- Usa menor potencia, múltiples pasadas
- Asegúrate de que el material esté plano y asegurado

**Para un grabado más rápido:**

- Usa un intervalo de línea más grande (0.15-0.2mm)
- Overscan mínimo (1-2mm)
- Pasada única a mayor potencia

### Problemas Comunes

**Marcas de quemadura al final de las líneas:**

- Aumenta la distancia de overscan
- Revisa los ajustes de aceleración
- Reduce ligeramente la potencia

**Líneas de escaneo visibles:**

- Disminuye el intervalo de línea
- Reduce la potencia (sobre-quemar crea huecos)
- Verifica que el material esté plano

**Grabado desigual:**

- Asegúrate de que el material esté plano
- Revisa la consistencia del enfoque
- Verifica la estabilidad de la potencia del láser
- Limpia la lente del láser

**Bandas (rayas oscuras/claras):**

- Calibra el [Desplazamiento de Escaneo Bidireccional](#desplazamiento-de-escaneo-bidireccional)
- Revisa la tensión de las correas
- Reduce la velocidad
- Prueba un ángulo de escaneo diferente

## Solución de Problemas

### Grabado muy ligero

- **Aumenta:** Configuración de potencia
- **Disminuye:** Configuración de velocidad
- **Revisa:** El enfoque es correcto
- **Prueba:** Múltiples pasadas

### Grabado muy oscuro/quemando

- **Disminuye:** Configuración de potencia
- **Aumenta:** Configuración de velocidad
- **Aumenta:** Intervalo de línea
- **Revisa:** El material es apropiado

### Oscuridad inconsistente

- **Revisa:** El material está plano
- **Revisa:** La distancia de enfoque es consistente
- **Verifica:** El haz del láser está limpio
- **Prueba:** Un área diferente del material (la veta varía)

### La imagen se ve pixelada

- **Disminuye:** Intervalo de línea
- **Revisa:** Resolución de la imagen fuente
- **Prueba:** Intervalo de línea más pequeño (0.05mm)
- **Verifica:** La imagen no se está ampliando

### Líneas de escaneo visibles

- **Disminuye:** Intervalo de línea
- **Reduce:** Potencia (sobre-quemar crea huecos)
- **Prueba:** Ángulo de escaneo diferente
- **Asegúrate:** La superficie del material es suave

## Temas Relacionados

- **[Corte de Contorno](contour)** - Cortar contornos y formas
- **[Overscan](../overscan.md)** - Mejorando la calidad del grabado
- **[Cuadrícula de Prueba de Material](material-test-grid)** - Encontrar ajustes óptimos
- **[Flujo de Trabajo Multi-Capa](../multi-layer.md)** - Combinando el grabado con otras operaciones
