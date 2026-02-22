# Integración de Cámara

Rayforge soporta la integración de cámara USB para la alineación y posicionamiento preciso del material. La función de superposición de cámara te permite ver exactamente dónde tu láser cortará o grabará en el material, eliminando las conjeturas y reduciendo el desperdicio de material.

![Ajustes de Cámara](/screenshots/machine-camera.png)

## Resumen

La integración de cámara proporciona:

- **Superposición de video en vivo** en el lienzo mostrando tu material en tiempo real
- **Alineación de imagen** para calibrar la posición de la cámara relativa al láser
- **Posicionamiento visual** para colocar trabajos con precisión en materiales irregulares o pre-marcados
- **Vista previa del material** antes de ejecutar trabajos
- **Soporte para múltiples cámaras** para diferentes configuraciones de máquina

:::tip Casos de Uso

- Alinear cortes en materiales pre-impresos
- Trabajar con materiales de formas irregulares
- Posicionamiento preciso de grabados en objetos existentes
- Reducir cortes de prueba y desperdicio de material
  :::

---

## Configuración de Cámara

### Requisitos de Hardware

**Cámaras compatibles:**

- Cámaras web USB (más común)
- Cámaras integradas de laptop (si ejecutas Rayforge en una laptop cerca de la máquina)
- Cualquier cámara soportada por Video4Linux2 (V4L2) en Linux o DirectShow en Windows

**Configuración recomendada:**

- Cámara montada sobre el área de trabajo con vista clara del material
- Condiciones de iluminación consistentes
- Cámara posicionada para capturar el área de trabajo del láser
- Montaje seguro para prevenir el movimiento de la cámara

### Añadir una Cámara

1. **Conecta tu cámara** a tu computadora vía USB

2. **Abre Ajustes de Cámara:**
   - Navega a **Configuración Preferencias Cámara**
   - O usa el botón de la barra de herramientas de cámara

3. **Añade una nueva cámara:**
   - Haz clic en el botón "+" para añadir una cámara
   - Ingresa un nombre descriptivo (ej., "Cámara Superior", "Cámara Área de Trabajo")
   - Selecciona el dispositivo del menú desplegable
     - En Linux: `/dev/video0`, `/dev/video1`, etc.
     - En Windows: Cámara 0, Cámara 1, etc.

4. **Habilita la cámara:**
   - Activa el interruptor de habilitación de cámara
   - La transmisión en vivo debería aparecer en tu lienzo

5. **Ajusta los ajustes de la cámara:**
   - **Brillo:** Ajusta si el material está muy oscuro/claro
   - **Contraste:** Mejora la visibilidad de los bordes
   - **Transparencia:** Controla la opacidad de la superposición (20-50% recomendado)
   - **Balance de Blancos:** Auto o temperatura Kelvin manual

---

## Alineación de Cámara

La alineación de cámara calibra la relación entre los píxeles de la cámara y las coordenadas del mundo real, permitiendo el posicionamiento preciso.

### Por Qué es Necesaria la Alineación

La cámara ve el área de trabajo desde arriba, pero la imagen puede estar:

- Rotada relativa a los ejes de la máquina
- Escalada diferente en direcciones X e Y
- Distorsionada por la perspectiva de la lente

La alineación crea una matriz de transformación que mapea los píxeles de la cámara a las coordenadas de la máquina.

### Procedimiento de Alineación

1. **Abre el Diálogo de Alineación:**
   - Haz clic en el botón de alineación de cámara en la barra de herramientas
   - O ve a **Cámara Alinear Cámara**

2. **Coloca marcadores de alineación:**
   - Necesitas al menos 3 puntos de referencia (4 recomendados para mejor precisión)
   - Los puntos de alineación deben estar distribuidos por el área de trabajo
   - Usa posiciones conocidas como:
     - Posición de origen de la máquina
     - Marcas de regla
     - Agujeros de alineación pre-cortados
     - Cuadrícula de calibración

3. **Marca puntos de imagen:**
   - Haz clic en la imagen de la cámara para colocar un punto en una ubicación conocida
   - El widget de burbuja aparece mostrando las coordenadas del punto
   - Repite para cada punto de referencia

4. **Ingresa coordenadas del mundo:**
   - Para cada punto de imagen, ingresa las coordenadas X/Y del mundo real en mm
   - Estas son las coordenadas reales de la máquina donde está ubicado cada punto
   - Mide con precisión con una regla o usa posiciones conocidas de la máquina

5. **Aplica la alineación:**
   - Haz clic en "Aplicar" para calcular la transformación
   - La superposición de la cámara ahora estará correctamente alineada

6. **Verifica la alineación:**
   - Mueve la cabeza del láser a una posición conocida
   - Verifica que el punto del láser se alinee con la posición esperada en la vista de la cámara
   - Ajusta volviendo a alinear si es necesario

### Consejos de Alineación

:::tip Mejores Prácticas
- Usa puntos en las esquinas de tu área de trabajo para máxima cobertura
- Evita agrupar puntos en una área
- Mide las coordenadas del mundo cuidadosamente - la precisión aquí determina la calidad general de la alineación
- Vuelve a alinear si mueves la cámara o cambias la distancia de enfoque
- Guarda tu alineación - persiste entre sesiones
  :::

**Flujo de trabajo de alineación de ejemplo:**

1. Mueve el láser a la posición de origen (0, 0) y marca en la cámara
2. Mueve el láser a (100, 0) y marca en la cámara
3. Mueve el láser a (100, 100) y marca en la cámara
4. Mueve el láser a (0, 100) y marca en la cámara
5. Ingresa las coordenadas exactas para cada punto
6. Aplica y verifica

---

## Usando la Superposición de Cámara

Una vez alineada, la superposición de cámara ayuda a posicionar trabajos con precisión.

### Habilitando/Deshabilitando la Superposición

- **Alternar cámara:** Haz clic en el ícono de cámara en la barra de herramientas
- **Ajustar transparencia:** Usa el deslizador en ajustes de cámara (20-50% funciona bien)
- **Refrescar imagen:** La cámara se actualiza continuamente mientras está habilitada

### Posicionando Trabajos con la Cámara

**Flujo de trabajo para posicionamiento preciso:**

1. **Habilita la superposición de cámara** para ver tu material

2. **Importa tu diseño** (SVG, DXF, etc.)

3. **Posiciona el diseño** en el lienzo:
   - Arrastra el diseño para alinear con las características visibles en la cámara
   - Usa zoom para ver detalles finos
   - Rota/escala según sea necesario

4. **Previsualiza la alineación:**
   - Usa el [Modo Simulación](../features/simulation-mode) para visualizar
   - Verifica que los cortes/grabados estén donde esperas

5. **Enmarca el trabajo** para verificar el posicionamiento antes de ejecutar

6. **Ejecuta el trabajo** con confianza

### Ejemplo: Grabando en una Tarjeta Pre-Impresa

1. Coloca la tarjeta impresa en la cama láser
2. Habilita la superposición de cámara
3. Importa tu diseño de grabado
4. Arrastra y posiciona el diseño para alinear con las características impresas
5. Ajusta la posición usando las teclas de flecha
6. Enmarca para verificar
7. Ejecuta el trabajo

---

## Referencia de Ajustes de Cámara

### Ajustes de Dispositivo

| Ajuste         | Descripción                     | Valores                               |
| -------------- | ------------------------------- | ------------------------------------- |
| **Nombre**     | Nombre descriptivo para la cámara | Cualquier texto                       |
| **ID de Dispositivo** | Identificador de dispositivo del sistema | `/dev/video0` (Linux), `0` (Windows)  |
| **Habilitado** | Estado activo de la cámara      | On/Off                                |

### Ajuste de Imagen

| Ajuste              | Descripción                   | Rango                               |
| ------------------- | ----------------------------- | ----------------------------------- |
| **Brillo**          | Brillo general de la imagen   | -100 a +100                         |
| **Contraste**       | Definición de bordes y contraste | 0 a 100                           |
| **Transparencia**   | Opacidad de la superposición en el lienzo | 0% (opaco) a 100% (transparente) |
| **Balance de Blancos** | Corrección de temperatura de color | Auto o 2000-10000K                |

### Datos de Alineación

| Propiedad                  | Descripción                         |
| ------------------------- | ----------------------------------- |
| **Puntos de Imagen**      | Coordenadas de píxel en imagen de cámara |
| **Puntos del Mundo**      | Coordenadas de máquina del mundo real (mm) |
| **Matriz de Transformación** | Mapeo calculado (interno)       |

---

## Funciones Avanzadas

### Calibración de Cámara (Corrección de Distorsión de Lente)

Para trabajo preciso, puedes calibrar la cámara para corregir la distorsión de barril/cojín:

1. **Imprime un patrón de tablero de ajedrez** (ej., cuadrícula de 8×6 con cuadrados de 25mm)
2. **Captura 10+ imágenes** del patrón desde diferentes ángulos/posiciones
3. **Usa herramientas de calibración OpenCV** para calcular la matriz de cámara y los coeficientes de distorsión
4. **Aplica la calibración** en Rayforge (ajustes avanzados)

:::note Cuándo Calibrar
La corrección de distorsión de lente solo es necesaria para:

- Lentes gran angular con distorsión de barril notable
- Trabajo de precisión que requiere <1mm de exactitud
- Áreas de trabajo grandes donde la distorsión se acumula

La mayoría de las cámaras web estándar funcionan bien sin calibración para trabajo láser típico.
:::

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

- Revisa el Administrador de Dispositivos para la cámara en "Cámaras" o "Dispositivos de imagen"
- Asegúrate de que ninguna otra aplicación esté usando la cámara (cierra Zoom, Skype, etc.)
- Prueba un puerto USB diferente
- Actualiza los controladores de la cámara

### La Cámara Muestra Pantalla Negra

**Problema:** La cámara es detectada pero no muestra imagen.

**Posibles causas:**

1. **Cámara en uso por otra aplicación** - Cierra otras aplicaciones de video
2. **Dispositivo incorrecto seleccionado** - Prueba diferentes IDs de dispositivo
3. **Permisos de cámara** - En Linux Snap, asegúrate de que la interfaz de cámara esté conectada
4. **Problema de hardware** - Prueba la cámara con otra aplicación

**Soluciones:**

```bash
# Linux: Liberar dispositivo de cámara
sudo killall cheese  # u otras aplicaciones de cámara

# Verificar qué proceso está usando la cámara
sudo lsof /dev/video0
```

### Alineación No Precisa

**Problema:** La superposición de cámara no coincide con la posición real del láser.

**Diagnóstico:**

1. **Puntos de alineación insuficientes** - Usa al menos 4 puntos
2. **Errores de medición** - Verifica doblemente las coordenadas del mundo
3. **Cámara movida** - Vuelve a alinear si la posición de la cámara cambió
4. **Distorsión no lineal** - Puede necesitar calibración de lente

**Mejora la precisión:**

- Usa más puntos de alineación (6-8 para áreas muy grandes)
- Distribuye los puntos por toda el área de trabajo
- Mide las coordenadas del mundo muy cuidadosamente
- Usa comandos de movimiento de la máquina para posicionar el láser con precisión en coordenadas conocidas
- Vuelve a alinear después de cualquier ajuste de la cámara

### Calidad de Imagen Pobre

**Problema:** La imagen de la cámara está borrosa, oscura o deslavada.

**Soluciones:**

1. **Ajusta brillo/contraste** en ajustes de cámara
2. **Mejora la iluminación** - Añade iluminación consistente del área de trabajo
3. **Limpia la lente de la cámara** - El polvo y escombros reducen la claridad
4. **Revisa el enfoque** - El autoenfoque puede no funcionar bien; usa manual si es posible
5. **Reduce la transparencia** temporalmente para ver la imagen de la cámara más claramente
6. **Prueba diferentes ajustes** de balance de blancos

### Retraso o Tartamudeo de la Cámara

**Problema:** La transmisión de cámara en vivo es entrecortada o retrasada.

**Soluciones:**

- Reduce la resolución de la cámara en ajustes del dispositivo (si es accesible)
- Cierra otras aplicaciones que usen CPU/GPU
- Actualiza los controladores de gráficos
- En Linux, asegúrate de usar el backend V4L2 (automático en Rayforge)
- Deshabilita la cámara cuando no se necesite para ahorrar recursos

---

## Páginas Relacionadas

- [Modo Simulación](../features/simulation-mode) - Previsualizar ejecución con superposición de cámara
- [Vista Previa 3D](../ui/3d-preview) - Visualizar trabajos en 3D
- [Enmarcando Trabajos](../features/framing-your-job) - Verificar posición del trabajo
- [Ajustes Generales](general) - Configuración de máquina
