# Grabado

Las operaciones de grabado rellenan áreas con líneas de escaneo rasterizado, soportando múltiples modos para diferentes efectos de grabado. Desde fotos en escala de grises suaves hasta efectos de relieve 3D, elige el modo que mejor se adapte a tu diseño y material.

## Resumen

Las operaciones de grabado:

- Rellenan formas cerradas con líneas de escaneo
- Soportan múltiples modos de grabado para diferentes efectos
- Funcionan tanto con formas vectoriales como imágenes de mapa de bits
- Usan escaneo bidireccional para velocidad
- Crean marcas permanentes en muchos materiales

## Modos de Grabado

### Modo de Potencia Variable

El modo de Potencia Variable varía la potencia del láser continuamente basándose en el brillo de la imagen, creando grabado en escala de grises suave con transiciones graduales.

**Mejor Para:**

- Fotos e imágenes en escala de grises suaves
- Degradados y transiciones naturales
- Retratos y obras de arte
- Grabado en madera y cuero

**Características Clave:**

- Modulación continua de potencia
- Control de potencia mín/máx
- Degradados suaves
- Mejor calidad tonal que el tramado

### Modo de Potencia Constante

El modo de Potencia Constante graba a potencia completa, con un umbral que determina qué píxeles se graban. Esto crea resultados limpios en blanco/negro.

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

El modo trama convierte imágenes en escala de grises a patrones binarios usando algoritmos de tramado, permitiendo grabado de fotos de alta calidad con mejor reproducción tonal que métodos simples basados en umbral.

**Mejor Para:**

- Grabar fotografías en madera o cuero
- Crear obras de arte estilo media tinta
- Imágenes con degradados suaves
- Cuando el rasterizado estándar no captura suficiente detalle

**Características Clave:**

- Múltiples opciones de algoritmos de tramado
- Mejor preservación de detalle
- Tonos continuos percibidos
- Ideal para fotografías

### Modo Múltiples Profundidades

El modo Múltiples Profundidades crea efectos de relieve 3D variando la potencia del láser basándose en el brillo de la imagen, con múltiples pasadas para tallado más profundo.

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
- Múltiples pasadas para grabado más profundo
- Escalonamiento Z entre pasadas

## Cuándo Usar Grabado

Usa operaciones de grabado para:

- Grabar texto y logos
- Crear imágenes y fotos en madera/cuero
- Rellenar áreas sólidas con textura
- Marcar partes y productos
- Crear efectos de relieve 3D
- Obras de arte estilo media tinta

**No uses grabado para:**

- Cortar a través del material (usa [Contorno](contour) en su lugar)
- Contornos precisos (el rasterizado crea áreas rellenas)
- Trabajo de líneas finas (los vectores son más limpios)

## Creando una Operación de Grabado

### Paso 1: Preparar Contenido

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

- **Potencia Variable** - Grabado en escala de grises suave
- **Potencia Constante** - Grabado limpio en blanco/negro
- **Trama** - Grabado de fotos de alta calidad
- **Múltiples Profundidades** - Efectos de relieve 3D

### Paso 4: Configurar Ajustes

![Ajustes de paso de grabado](/screenshots/step-settings-engrave-general-variable.png)

## Ajustes Comunes

### Potencia y Velocidad

**Potencia (%):**

- Intensidad del láser para grabar
- Potencia más baja para marcado más ligero
- Potencia más alta para grabado más profundo

**Velocidad (mm/min):**

- Qué tan rápido escanea el láser
- Más rápido = más ligero, más lento = más oscuro

### Intervalo de Línea

**Intervalo de Línea (mm):**

- Espaciado entre líneas de escaneo
- Menor = mayor calidad, tiempo de trabajo más largo
- Mayor = más rápido, líneas visibles

| Intervalo | Calidad | Velocidad | Usar Para               |
| --------- | ------- | --------- | ----------------------- |
| 0.05mm    | Máxima  | Más lento | Fotos, detalle fino     |
| 0.1mm     | Alta    | Media     | Texto, logos, gráficos  |
| 0.2mm     | Media   | Rápido    | Rellenos sólidos, texturas |
| 0.3mm+    | Baja    | Más rápido| Borrador, pruebas       |

**Recomendado:** 0.1mm para uso general

:::tip Coincidencia de Resolución
:::

Para imágenes, el intervalo de línea debería coincidir o exceder la resolución de la imagen. Si tu imagen es 10 píxeles/mm (254 DPI), usa intervalo de línea de 0.1mm o menor.

### Dirección de Escaneo

**Ángulo de Escaneo (grados):**

- Dirección de las líneas de escaneo
- 0 = horizontal (izquierda a derecha)
- 90 = vertical (arriba a abajo)
- 45 = diagonal

**¿Por qué cambiar el ángulo?**

- Veta de la madera: Graba perpendicular a la veta para mejores resultados
- Orientación del patrón: Coincidir con la estética del diseño
- Reducir bandas: Diferente ángulo puede ocultar imperfecciones

**Escaneo Bidireccional:**

- **Habilitado:** El láser graba en ambas direcciones (más rápido)
- **Deshabilitado:** El láser solo graba de izquierda a derecha (más lento, más consistente)

Para mejor calidad, deshabilita bidireccional. Para velocidad, habilítalo.

### Overscan

**Distancia de Overscan (mm):**

- Qué tan más allá del diseño viaja el láser antes de dar la vuelta
- Permite que el láser alcance velocidad completa antes de entrar al diseño
- Previene marcas de quemadura al inicio/final de líneas

**Valores típicos:**

- 2-5mm para la mayoría de trabajos
- Mayor para velocidades altas

Ver [Overscan](../overscan) para detalles.

## Ajustes Específicos del Modo

### Ajustes del Modo Potencia Variable

![Ajustes del modo Potencia Variable](/screenshots/step-settings-engrave-general-variable.png)

**Potencia Mín (%):**

- Potencia del láser para áreas más claras (píxeles blancos)
- Usualmente 0-20%
- Establecer más alto para evitar áreas muy superficiales

**Potencia Máx (%):**

- Potencia del láser para áreas más oscuras (píxeles negros)
- Usualmente 40-80% dependiendo del material
- Menor = relieve sutil, mayor = profundidad dramática

**Ejemplos de Rango de Potencia:**

| Mín | Máx | Efecto                |
| --- | --- | --------------------- |
| 0%  | 40% | Relieve sutil, ligero |
| 10% | 60% | Profundidad media, seguro |
| 20% | 80% | Profundo, relieve dramático |

**Invertir:**

- **Apagado** (por defecto): Blanco = superficial, Negro = profundo
- **Encendido**: Blanco = profundo, Negro = superficial

Usa invertir para litofanías (áreas claras deberían ser delgadas) o repujado (áreas elevadas).

**Rango de Brillo:**

Controla cómo se mapean los valores de brillo de la imagen a la potencia del láser. El histograma muestra la distribución de valores de brillo en tu imagen.

- **Auto Niveles** (por defecto): Ajusta automáticamente los puntos de negro y blanco basándose en el contenido de la imagen. Valores por debajo del punto de negro se tratan como negro, valores por encima del punto blanco se tratan como blanco. Esto estira el contraste de la imagen para usar el rango completo de potencia.
- **Modo Manual**: Deshabilita Auto Niveles para establecer manualmente los puntos de negro y blanco arrastrando los marcadores en el histograma.

Esto es particularmente útil para:
- Imágenes de bajo contraste que necesitan mejora de contraste
- Imágenes con rango tonal limitado
- Asegurar resultados consistentes a través de diferentes imágenes fuente

### Ajustes del Modo Potencia Constante

![Ajustes del modo Potencia Constante](/screenshots/step-settings-engrave-general-constant_power.png)

**Umbral (0-255):**

- Corte de brillo para separación blanco/negro
- Menor = más negro grabado
- Mayor = más blanco grabado

**Valores típicos:**

- 128 (umbral 50% gris)
- Ajustar basándose en el contraste de la imagen

### Ajustes del Modo Trama

![Ajustes del modo Trama](/screenshots/step-settings-engrave-general-dither.png)

**Algoritmo de Tramado:**

Elige el algoritmo que mejor se adapte a tu imagen y material:

| Algoritmo        | Calidad | Velocidad | Mejor Para                            |
| ---------------- | ------- | --------- | ------------------------------------- |
| Floyd-Steinberg  | Máxima  | Más lento | Fotos, retratos, degradados suaves    |
| Bayer 2x2        | Baja    | Más rápido| Efecto media tonta grueso             |
| Bayer 4x4        | Media   | Rápido    | Media tonta equilibrada               |
| Bayer 8x8        | Alta    | Media     | Detalle fino, patrones sutiles        |

**Floyd-Steinberg** es el predeterminado y recomendado para la mayoría de grabados de fotos. Usa difusión de error para distribuir errores de cuantización a píxeles vecinos, creando resultados de aspecto natural.

**Tramado Bayer** crea patrones regulares que pueden producir efectos artísticos que asemejan la impresión tradicional de media tinta.

### Ajustes del Modo Múltiples Profundidades

![Ajustes del modo Múltiples Profundidades](/screenshots/step-settings-engrave-general-multi_pass.png)

**Número de Niveles de Profundidad:**

- Número de niveles de profundidad discretos
- Más niveles = degradados más suaves
- Típico: 5-10 niveles

**Paso-Z por Nivel (mm):**

- Cuánto bajar entre pasadas de profundidad
- Crea profundidad total más profunda con múltiples pasadas
- Típico: 0.1-0.5mm

**Rotar Ángulo Por Pasada:**

- Grados para rotar cada pasada sucesiva
- Crea efecto 3D tipo entramado cruzado
- Típico: 0-45 grados

**Invertir:**

- **Habilitado:** Blanco = profundo, Negro = superficial
- **Deshabilitado:** Negro = profundo, Blanco = superficial

Usa invertir para litofanías (áreas claras deberían ser delgadas) o repujado (áreas elevadas).

## Consejos y Mejores Prácticas

![Ajustes de post-procesamiento de grabado](/screenshots/step-settings-engrave-post.png)

### Selección de Material

**Mejores materiales para grabar:**

- Madera (las variaciones naturales crean resultados hermosos)
- Cuero (se quema a marrón oscuro/negro)
- Aluminio anodizado (remueve el recubrimiento, revela el metal)
- Metales recubiertos (remueve la capa de recubrimiento)
- Algunos plásticos (¡prueba primero!)

**Materiales desafiantes:**

- Acrílico transparente (no muestra bien el grabado)
- Metales sin recubrimiento (requiere compuestos de marcado especiales)
- Vidrio (requiere ajustes/recubrimientos especiales)

### Ajustes de Calidad

**Para mejor calidad:**

- Usa intervalo de línea más pequeño (0.05-0.1mm)
- Deshabilita escaneo bidireccional
- Aumenta overscan (3-5mm)
- Usa potencia más baja, múltiples pasadas
- Asegúrate de que el material esté plano y asegurado

**Para grabado más rápido:**

- Usa intervalo de línea más grande (0.15-0.2mm)
- Habilita escaneo bidireccional
- Overscan mínimo (1-2mm)
- Pasada única a potencia más alta

### Problemas Comunes

**Marcas de quemadura al final de líneas:**

- Aumenta la distancia de overscan
- Revisa los ajustes de aceleración
- Reduce la potencia ligeramente

**Líneas de escaneo visibles:**

- Disminuye el intervalo de línea
- Reduce la potencia (sobre-quemar crea huecos)
- Verifica que el material esté plano

**Grabado desigual:**

- Asegúrate de que el material esté plano
- Revisa la consistencia del enfoque
- Verifica la estabilidad de potencia del láser
- Limpia la lente del láser

**Bandas (rayas oscuras/claras):**

- Deshabilita escaneo bidireccional
- Revisa la tensión de las correas
- Reduce la velocidad
- Prueba diferente ángulo de escaneo

## Solución de Problemas

### Grabado muy ligero

- **Aumenta:** Ajuste de potencia
- **Disminuye:** Ajuste de velocidad
- **Revisa:** El enfoque es correcto
- **Prueba:** Múltiples pasadas

### Grabado muy oscuro/quemando

- **Disminuye:** Ajuste de potencia
- **Aumenta:** Ajuste de velocidad
- **Aumenta:** Intervalo de línea
- **Revisa:** El material es apropiado

### Oscuridad inconsistente

- **Revisa:** El material está plano
- **Revisa:** La distancia de enfoque es consistente
- **Verifica:** El haz del láser está limpio
- **Prueba:** Diferente área del material (la veta varía)

### La imagen se ve pixelada

- **Disminuye:** Intervalo de línea
- **Revisa:** Resolución de la imagen fuente
- **Prueba:** Intervalo de línea más pequeño (0.05mm)
- **Verifica:** La imagen no se está escalando hacia arriba

### Líneas de escaneo visibles

- **Disminuye:** Intervalo de línea
- **Reduce:** Potencia (sobre-quemar crea huecos)
- **Prueba:** Diferente ángulo de escaneo
- **Asegúrate:** La superficie del material es suave

## Temas Relacionados

- **[Corte de Contorno](contour)** - Cortar contornos y formas
- **[Overscan](../overscan)** - Mejorando la calidad del grabado
- **[Cuadrícula de Prueba de Materiales](material-test-grid)** - Encontrar ajustes óptimos
- **[Flujo de Trabajo Multi-Capa](../multi-layer)** - Combinando grabado con otras operaciones
