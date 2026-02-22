# Overscan

Overscan extiende las líneas de grabado raster más allá del área de contenido real para asegurar que el láser alcance velocidad constante durante el grabado, eliminando artefactos de aceleración.

## El Problema: Marcas de Aceleración

Sin overscan, el grabado raster sufre de **artefactos de aceleración**:

- **Bordes claros** donde la aceleración comienza (láser moviéndose muy rápido para el nivel de potencia)
- **Bordes oscuros** donde ocurre la desaceleración (láser permaneciendo más tiempo)
- **Profundidad/oscuridad de grabado inconsistente** a lo largo de cada línea
- Bandas o rayas visibles en los bordes de las líneas

## Cómo Funciona Overscan

Overscan **extiende la trayectoria de herramienta** antes y después de cada línea raster:

**Proceso:**

1. **Entrada:** El láser se mueve a una posición _antes_ de que comience la línea
2. **Acelerar:** El láser acelera a la velocidad objetivo (láser APAGADO)
3. **Grabar:** El láser se enciende y graba a velocidad constante
4. **Desacelerar:** El láser se apaga y desacelera _después_ de que termina la línea

**Resultado:** Toda el área grabada recibe potencia consistente a velocidad constante.

**Beneficios:**

- Profundidad de grabado uniforme a lo largo de toda la línea raster
- Sin bordes claros/oscuros
- Mayor calidad en grabado de fotos
- Resultados de apariencia profesional

## Configurando Overscan

Overscan es un **transformador** en el flujo de trabajo pipeline de Rayforge.

**Para habilitar:**

1. **Selecciona la capa** con grabado raster
2. **Abre configuración de flujo de trabajo** (o configuración de operación)
3. **Añade transformador Overscan** si no está ya presente
4. **Configura la distancia**

**Ajustes:**

| Ajuste             | Descripción                 | Valor Típico    |
| ------------------ | --------------------------- | --------------- |
| **Habilitado**     | Alternar overscan on/off    | ON (para raster)|
| **Distancia (mm)** | Cuánto extender las líneas  | 2-5 mm          |

## Elegir la Distancia de Overscan

La distancia de overscan debería permitir que la máquina **acelere completamente** a la velocidad objetivo.

**Directrices prácticas:**

| Velocidad Máxima          | Aceleración | Overscan Recomendado |
| ------------------------- | ----------- | -------------------- |
| 3000 mm/min (50 mm/s)     | Baja        | 5 mm                 |
| 3000 mm/min (50 mm/s)     | Media       | 3 mm                 |
| 3000 mm/min (50 mm/s)     | Alta        | 2 mm                 |
| 6000 mm/min (100 mm/s)    | Baja        | 10 mm                |
| 6000 mm/min (100 mm/s)    | Media       | 6 mm                 |
| 6000 mm/min (100 mm/s)    | Alta        | 4 mm                 |

**Factores que afectan la distancia requerida:**

- **Velocidad:** Mayor velocidad = necesita más distancia para acelerar
- **Aceleración:** Menor aceleración = necesita más distancia
- **Mecánica de máquina:** Transmisión por correa vs transmisión directa afecta la aceleración

**Ajuste:**

- **Muy poco:** Marcas de aceleración aún visibles en los bordes
- **Demasiado:** Pierde tiempo, puede exceder límites de la máquina
- **Comienza con 3mm** y ajusta basado en resultados

## Probando Ajustes de Overscan

**Procedimiento de prueba:**

1. **Crea un grabado de prueba:**
   - Rectángulo relleno sólido (50mm x 20mm)
   - Usa tus ajustes de grabado típicos
   - Habilita overscan a 3mm

2. **Graba la prueba:**
   - Ejecuta el trabajo
   - Permite que complete

3. **Examina los bordes:**
   - Mira los bordes izquierdo y derecho del rectángulo
   - Busca variación de oscuridad en los bordes
   - Compara la oscuridad del borde con la oscuridad del centro

4. **Ajusta:**
   - **Si los bordes son más claros/oscuros:** Aumenta overscan
   - **Si los bordes coinciden con el centro:** Overscan es suficiente
   - **Si los bordes son perfectos:** Intenta reducir overscan ligeramente para ahorrar tiempo

## Cuándo Usar Overscan

**Siempre usa para:**

- Grabado de fotos (raster)
- Patrones de relleno
- Cualquier trabajo raster de alto detalle
- Grabado de imágenes en escala de grises
- Grabado de texto (modo raster)

**Opcional para:**

- Corte vectorial (no necesario)
- Grabado muy lento (aceleración menos notable)
- Formas simples grandes (bordes menos críticos)

**Deshabilita para:**

- Operaciones vectoriales
- Áreas de trabajo muy pequeñas (puede exceder límites)
- Cuando la calidad del borde no es importante

---

## Temas Relacionados

- [Operaciones de Grabado](./operations/engrave) - Configurar ajustes de grabado
- [Cuadrícula de Prueba de Material](./operations/material-test-grid) - Encontrar potencia/velocidad óptimos
