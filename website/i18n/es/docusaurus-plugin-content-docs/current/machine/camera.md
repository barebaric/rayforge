---
description: "Configure la calibración de la cámara en Rayforge para la alineación precisa de la pieza de trabajo. Use su cámara para previsualizar y posicionar diseños en materiales."
---

# Integración de Cámara

Rayforge soporta la integración de cámara USB para la alineación y
posicionamiento preciso del material. La función de superposición de cámara te
permite ver exactamente dónde tu láser va a cortar o grabar en el material,
eliminando las conjeturas y reduciendo el desperdicio de material.

![Ajustes de Cámara](/screenshots/machine-camera.png)

## Flujo de trabajo de configuración

La configuración de una cámara sigue cuatro pasos:

1. **Añadir una cámara** — Conecta tu cámara y agrégala a la configuración de
   la máquina
2. **Ajustar la configuración de imagen** — Ajusta brillo, contraste, balance
   de blancos y reducción de ruido
3. **Calibrar la lente** — Corrige la distorsión con el asistente de
   calibración o coeficientes manuales
4. **Alinear la cámara** — Mapa los píxeles de la cámara a las coordenadas de
   la máquina para un posicionamiento preciso

Los pasos 2–4 se acceden desde el panel de propiedades de la cámara, donde los
íconos de estado muestran el progreso de un vistazo:

- ✓ **Calibración de lente** — La calibración se ha realizado
- ⚠ **Alineación de imagen** — Advertencia cuando la alineación debe rehacerse
  (p. ej., después de la calibración de lente)
- ✓ **Alineación de imagen** — La alineación está actualizada y es válida

---

## Paso 1: Añadir una cámara

### Requisitos de Hardware

**Cámaras compatibles:**

- Cámaras web USB (más común)
- Cámaras integradas de laptop (si ejecutas Rayforge en una laptop cerca
  de la máquina)
- Cualquier cámara soportada por Video4Linux2 (V4L2) en Linux o
  DirectShow en Windows

**Configuración recomendada:**

- Cámara montada sobre el área de trabajo con vista clara del material
- Condiciones de iluminación consistentes
- Cámara posicionada para capturar el área de trabajo del láser
- Montaje seguro para prevenir el movimiento de la cámara

### Añadir una Cámara

1. **Conecta tu cámara** a tu computadora vía USB

2. **Abre Ajustes de Cámara:**
   - Navega a **Configuración → Preferencias → Cámara**
   - O usa el botón de la barra de herramientas de cámara

3. **Añade una nueva cámara:**
   - Haz clic en el botón "+" para añadir una cámara
   - Ingresa un nombre descriptivo (ej., "Cámara Superior",
     "Cámara Área de Trabajo")
   - Selecciona el dispositivo del menú desplegable
     - En Linux: `/dev/video0`, `/dev/video1`, etc.
     - En Windows: Cámara 0, Cámara 1, etc.

4. **Habilita la cámara:**
   - Activa el interruptor de habilitación de cámara
   - La transmisión en vivo debería aparecer en tu lienzo

---

## Paso 2: Ajustar la configuración de imagen

![Diálogo de Configuración de Imagen](/screenshots/camera-image-settings.png)

Haz clic en **Configurar** junto a **Ajustes de Imagen** en las propiedades de
la cámara para abrir el diálogo de configuración de imagen. Ajusta estos
parámetros para obtener una vista de cámara clara:

| Ajuste                 | Descripción                                                                               |
| ---------------------- | ----------------------------------------------------------------------------------------- |
| **Brillo**             | Brillo general de la imagen (-100 a +100)                                                 |
| **Contraste**          | Definición de bordes y contraste (0 a 100)                                                |
| **Preferir YUYV**      | Usar YUYV sin comprimir en lugar de MJPEG. Más lento pero puede solucionar algunos fallos |
| **Transparencia**      | Opacidad de la superposición en el lienzo (0% opaco a 100% transparente)                  |
| **Balance de Blancos** | Corrección de temperatura de color (Auto o 2500-10000K)                                   |
| **Reducción de Ruido** | Reducción de ruido temporal (0.0 a 0.95)                                                  |

La opción YUYV es útil si tu cámara produce imágenes con tono verdoso con el
formato MJPEG predeterminado. Ten en cuenta que YUYV no está comprimido y
puede reducir la resolución disponible o la tasa de fotogramas en conexiones
USB 2.0.

---

## Paso 3: Calibración de lente

Si tu cámara tiene una lente gran angular o está montada en ángulo, la
imagen puede mostrar curvatura visible — las líneas rectas aparecen
dobladas, especialmente cerca de los bordes del encuadre. Esto se llama
distorsión de lente, y puede afectar la alineación incluso si tus puntos
de alineación están medidos con cuidado.

Rayforge incluye un asistente de calibración guiado que corrige esta
distorsión automáticamente. También puedes ajustar los coeficientes de
distorsión manualmente.

### Diálogo de Calibración de Lente

![Diálogo de Calibración de Lente](/screenshots/camera-lens-calibration.png)

Abre el diálogo de calibración de lente haciendo clic en **Configurar**
junto a **Calibración de Lente** en las propiedades de la cámara. Desde
aquí puedes:

- **Ajustar coeficientes de distorsión manualmente** — Ajusta finamente
  los parámetros de distorsión radial (k1–k3) y tangencial (p1–p2)
- **Iniciar el asistente de calibración** — Haz clic en el botón
  **Asistente** para una calibración automática guiada

Los ajustes manuales son útiles para el ajuste fino después de que el
asistente haya calculado una solución inicial, o cuando conoces los
valores de distorsión aproximados para tu lente.

### Asistente de Calibración

El asistente de calibración te guía para capturar varias imágenes de una
tarjeta de calibración impresa desde diferentes posiciones en la cama.
Luego calcula un modelo de distorsión automáticamente.

**Paso 1: Configurar la tarjeta de calibración**

![Asistente — Configuración de
Tarjeta](/screenshots/camera-lens-calibration-wizard-card.png)

1. Haz clic en **Asistente** en el diálogo de calibración de lente para
   comenzar
2. Establece el **Ancho** y **Alto** de tu tarjeta impresa
3. La vista previa se actualiza en tiempo real — la tarjeta debe cubrir
   aproximadamente el 70% de la vista de la cámara
4. Haz clic en **Guardar como PDF** para exportar la tarjeta para imprimir
5. Imprime la tarjeta y colócala en la cama láser

**Paso 2: Capturar fotogramas**

![Asistente — Captura](/screenshots/camera-lens-calibration-wizard-capture.png)

1. Haz clic en **Siguiente** para entrar al modo de captura
2. Posiciona la tarjeta de calibración en diferentes ubicaciones y
   ángulos dentro de la vista de la cámara
3. Haz clic en **Capturar Fotograma** para cada posición
4. Apunta a al menos 8 capturas que cubran todo el encuadre, incluyendo
   esquinas y bordes
5. La barra de progreso y los indicadores de estado muestran la calidad
   de captura

**Paso 3: Aplicar calibración**

1. Una vez que se hayan capturado suficientes fotogramas, haz clic en
   **Calibrar**
2. Los coeficientes de distorsión calculados se aplican automáticamente
   a la cámara
3. La superposición de la cámara ahora muestra una imagen corregida y recta

---

## Paso 4: Alineación de imagen

![Diálogo de Alineación de Imagen](/screenshots/camera-image-alignment.png)

La alineación de cámara calibra la relación entre los píxeles de la cámara y
las coordenadas del mundo real, permitiendo el posicionamiento preciso.

### Por Qué es Necesaria la Alineación

La cámara ve el área de trabajo desde arriba, pero la imagen puede estar:

- Rotada relativa a los ejes de la máquina
- Escalada diferente en direcciones X e Y
- Distorsionada por la perspectiva de la lente

La alineación crea una matriz de transformación que mapea los píxeles de la
cámara a las coordenadas de la máquina.

### Procedimiento de Alineación

1. **Abre el Diálogo de Alineación:**
   - Haz clic en el botón **Configurar** junto a **Alineación de Imagen** en
     las propiedades de la cámara
   - El diálogo muestra la transmisión de la cámara con la superposición de
     alineación actual

2. **Coloca marcadores de alineación:**
   - Necesitas al menos 3 puntos de referencia (4 recomendados para mejor
     precisión)
   - Los puntos de alineación deben estar distribuidos por el área de trabajo
   - Usa posiciones conocidas como:
     - Posición de origen de la máquina
     - Marcas de regla
     - Agujeros de alineación pre-cortados
     - Cuadrícula de calibración

3. **Marca puntos de imagen:**
   - Haz clic en la imagen de la cámara para colocar un punto en una ubicación
     conocida
   - El widget de burbuja aparece mostrando las coordenadas del punto
   - Repite para cada punto de referencia

4. **Ingresa coordenadas del mundo:**
   - Para cada punto de imagen, ingresa las coordenadas X/Y del mundo real en
     mm
   - Estas son las coordenadas reales de la máquina donde está ubicado cada
     punto
   - Mide con precisión con una regla o usa posiciones conocidas de la máquina

5. **Aplica la alineación:**
   - Haz clic en **Aplicar** para calcular la transformación
   - La superposición de la cámara ahora estará correctamente alineada

6. **Verifica la alineación:**
   - Mueve la cabeza del láser a una posición conocida
   - Verifica que el punto del láser se alinee con la posición esperada en la
     vista de la cámara
   - Ajusta volviendo a alinear si es necesario

### Estado de Alineación

El panel de propiedades de la cámara muestra el estado de alineación con un
ícono:

- **Marca de verificación** — La alineación está actualizada y es válida
- **Advertencia** — La alineación debe rehacerse. Esto ocurre cuando se
  actualiza la calibración de lente, porque la corrección de distorsión cambia
  la imagen de la cámara e invalida la alineación existente. Tus puntos de
  alineación se conservan — simplemente abre el diálogo y haz clic en
  **Aplicar** nuevamente.

### Flujo de trabajo de ejemplo

1. Mueve el láser a la posición de origen (0, 0) y marca en la cámara
2. Mueve el láser a (100, 0) y marca en la cámara
3. Mueve el láser a (100, 100) y marca en la cámara
4. Mueve el láser a (0, 100) y marca en la cámara
5. Ingresa las coordenadas exactas para cada punto
6. Aplica y verifica

:::tip Mejores Prácticas

- Usa puntos en las esquinas de tu área de trabajo para máxima cobertura
- Evita agrupar puntos en una área
- Mide las coordenadas del mundo cuidadosamente - la precisión aquí
  determina la calidad general de
  la alineación
- Vuelve a alinear si mueves la cámara o cambias la distancia de enfoque
- Vuelve a alinear después de actualizar la calibración de lente
- Guarda tu alineación - persiste entre sesiones
  :::

---

## Usando la Superposición de Cámara

Una vez alineada, la superposición de cámara ayuda a posicionar trabajos
con precisión. Actívala haciendo clic en el icono de cámara en la barra
de herramientas de la ventana principal.

---

### Múltiples Cámaras

Rayforge soporta múltiples cámaras para diferentes vistas o máquinas:

- Añade múltiples cámaras en preferencias
- Cada cámara puede tener alineación independiente
- Cambia entre cámaras usando el selector de cámara
- Casos de uso:
  - Vista superior + vista lateral para objetos 3D
  - Diferentes cámaras para diferentes máquinas
  - Cámara gran angular + cámara de detalle

---

## Solución de Problemas

### Cámara No Detectada

**Problema:** La cámara no aparece en la lista de dispositivos.

**Soluciones:**

**Linux:**
Verifica si la cámara es reconocida por el sistema:

```bash
# Listar dispositivos de video
ls -l /dev/video*

# Verificar cámara con v4l2
v4l2-ctl --list-devices

# Probar con otra aplicación
cheese  # o VLC, etc.
```

**Para usuarios de Snap:**

```bash
# Conceder acceso a la cámara
sudo snap connect rayforge:camera
```

**Windows:**

- Revisa el Administrador de Dispositivos para la cámara en "Cámaras" o
  "Dispositivos de imagen"
- Asegúrate de que ninguna otra aplicación esté usando la cámara (cierra
  Zoom, Skype, etc.)
- Prueba un puerto USB diferente
- Actualiza los controladores de la cámara

### La Cámara Muestra Pantalla Negra

**Problema:** La cámara es detectada pero no muestra imagen.

**Posibles causas:**

1. **Cámara en uso por otra aplicación** - Cierra otras aplicaciones de video
2. **Dispositivo incorrecto seleccionado** - Prueba diferentes IDs de
   dispositivo
3. **Permisos de cámara** - En Linux Snap, asegúrate de que la interfaz
   de cámara esté conectada
4. **Problema de hardware** - Prueba la cámara con otra aplicación

**Soluciones:**

```bash
# Linux: Liberar dispositivo de cámara
sudo killall cheese  # u otras aplicaciones de cámara

# Verificar qué proceso está usando la cámara
sudo lsof /dev/video0
```

### Alineación No Precisa

**Problema:** La superposición de cámara no coincide con la posición
real del láser.

**Diagnóstico:**

1. **Puntos de alineación insuficientes** - Usa al menos 4 puntos
2. **Errores de medición** - Verifica doblemente las coordenadas del mundo
3. **Cámara movida** - Vuelve a alinear si la posición de la cámara cambió
4. **Distorsión no lineal** - Puede necesitar calibración de lente

**Mejora la precisión:**

- Usa más puntos de alineación (6-8 para áreas muy grandes)
- Distribuye los puntos por toda el área de trabajo
- Mide las coordenadas del mundo muy cuidadosamente
- Usa comandos de movimiento de la máquina para posicionar precisamente
  el láser en coordenadas conocidas
- Vuelve a alinear después de cualquier ajuste de la cámara

### Calidad de Imagen Pobre

**Problema:** La imagen de la cámara está borrosa, oscura o deslavada.

**Soluciones:**

1. **Ajusta brillo/contraste** en ajustes de cámara
2. **Mejora la iluminación** - Añade iluminación consistente del área de
   trabajo
3. **Limpia la lente de la cámara** - El polvo y escombros reducen la claridad
4. **Revisa el enfoque** - El autoenfoque puede no funcionar bien; usa
   manual si es posible
5. **Reduce la transparencia** temporalmente para ver la imagen de la
   cámara más claramente
6. **Prueba diferentes ajustes** de balance de blancos
7. **Ajusta la reducción de ruido** si la imagen aparece granulada

### Retraso o Tartamudeo de la Cámara

**Problema:** La transmisión de cámara en vivo es entrecortada o retrasada.

**Soluciones:**

- Reduce la resolución de la cámara en ajustes del dispositivo (si es
  accesible)
- Cierra otras aplicaciones que usen CPU/GPU
- Actualiza los controladores de gráficos

---

## Páginas Relacionadas

- [Vista Previa 3D](../ui/3d-preview) — Visualizar ejecución con superposición
  de cámara
- [Enmarcando Trabajos](../features/framing-your-job) — Verificar posición del
  trabajo
- [Ajustes Generales](general) — Configuración de máquina
