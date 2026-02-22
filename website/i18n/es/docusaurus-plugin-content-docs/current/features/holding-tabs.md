# Pestañas de Sujeción

Las pestañas de sujeción (también llamadas puentes o tabs) son pequeñas secciones sin cortar que se dejan a lo largo de las trayectorias de corte para mantener las piezas adjuntas al material circundante. Esto previene que las piezas cortadas se muevan durante el trabajo, lo que podría causar desalineación, daño o riesgos de incendio.

## ¿Por Qué Usar Pestañas de Sujeción?

Al cortar a través del material, la pieza cortada puede:

- **Cambiar de posición** a mitad del trabajo, causando que operaciones posteriores se desalineen
- **Caer a través** de la rejilla de la cama o inclinarse si solo está soportada en los bordes
- **Colisionar con** el cabezal láser mientras se mueve
- **Incendiarse** si cae sobre recortes calientes debajo
- **Dañarse** por caídas o vibración

Las pestañas de sujeción resuelven estos problemas manteniendo la pieza adjunta hasta que estés listo para removerla.

---

## Cómo Funcionan las Pestañas de Sujeción

Rayforge implementa pestañas creando **pequeños huecos en la trayectoria de corte**:

1. Marcas posiciones a lo largo de la trayectoria de corte donde deberían estar las pestañas
2. Durante la generación del G-code, Rayforge interrumpe el corte en cada pestaña
3. El láser se levanta (o se apaga), salta el ancho de la pestaña, luego reanuda el corte
4. Después de completar el trabajo, manualmente rompes o cortas las pestañas para liberar la pieza

---

## Añadir Pestañas de Sujeción

### Añadir Rápido

1. **Selecciona la pieza de trabajo** a la que quieres añadir pestañas (debe ser una operación de corte/contorno)
2. **Haz clic en la herramienta de pestañas** en la barra de herramientas o presiona el atajo de pestaña
3. **Haz clic en la trayectoria** donde quieres las pestañas:
   - Las pestañas aparecen como pequeñas manijas en el contorno de la trayectoria
   - Haz clic múltiples veces para añadir más pestañas
   - Típicamente 3-4 pestañas para piezas pequeñas, más para piezas más grandes
4. **Habilita las pestañas** si no están ya habilitadas (alternar en el panel de propiedades)

### Usando el Popover de Añadir Pestañas

Para más control:

1. **Clic derecho** en la pieza de trabajo o usa **Editar → Añadir Pestañas**
2. **Elige el método de colocación de pestañas:**
   - **Manual:** Haz clic en ubicaciones individuales
   - **Equidistante:** Espacia automáticamente las pestañas uniformemente alrededor de la trayectoria
3. **Configura los ajustes de pestañas:**
   - **Número de pestañas:** Cuántas pestañas crear (para equidistante)
   - **Ancho de pestaña:** Longitud de cada sección sin cortar (típicamente 2-5mm)
4. **Haz clic en Aplicar**

---

## Propiedades de Pestañas

### Ancho de Pestaña

El **ancho** es la longitud de la sección sin cortar a lo largo de la trayectoria.

**Anchos recomendados:**

| Material        | Espesor | Ancho de Pestaña |
| --------------- | ------- | ---------------- |
| **Cartón**      | 1-3mm   | 2-3mm            |
| **Contrachapado** | 3mm   | 3-4mm            |
| **Contrachapado** | 6mm   | 4-6mm            |
| **Acrílico**    | 3mm     | 2-3mm            |
| **Acrílico**    | 6mm     | 3-5mm            |
| **MDF**         | 3mm     | 3-4mm            |
| **MDF**         | 6mm     | 5-7mm            |

**Directrices:**
- **Materiales más gruesos** necesitan pestañas más anchas para resistencia
- **Partes más pesadas** necesitan más y/o pestañas más anchas
- **Materiales frágiles** (acrílico) pueden usar pestañas más pequeñas (más fáciles de romper)
- **Materiales fibrosos** (madera) pueden necesitar pestañas más anchas

:::warning Ancho de Pestaña vs Espesor de Material
Las pestañas deben ser lo suficientemente anchas para soportar la pieza pero lo suficientemente pequeñas para removerse limpiamente. Muy estrecha = la pieza puede liberarse; muy ancha = difícil de remover o daña la pieza.
:::

### Posición de Pestaña

Las pestañas se posicionan usando dos parámetros:

- **Índice de segmento:** Qué segmento de línea/arco de la trayectoria
- **Posición (0.0 - 1.0):** Dónde a lo largo de ese segmento (0 = inicio, 1 = final)

**Consejos de colocación manual:**
- Coloca pestañas en **secciones rectas** cuando sea posible (más fácil de remover)
- Evita pestañas en **curvas cerradas** (concentración de tensión)
- Distribuye pestañas **uniformemente** alrededor de la pieza
- Coloca pestañas en **esquinas** para máximo soporte si es necesario

### Pestañas Equidistantes

La función **equidistante** coloca automáticamente pestañas a intervalos uniformes:

**Beneficios:**
- Distribución de peso uniforme
- Patrón de rotura predecible
- Configuración rápida para formas regulares

---

## Trabajando con Pestañas

### Editar Pestañas

**Mover una pestaña:**
1. Selecciona la pieza de trabajo
2. Arrastra la manija de la pestaña a lo largo de la trayectoria
3. Suelta para establecer la nueva posición

**Redimensionar una pestaña:**
- Usa el panel de propiedades para ajustar el ancho
- Todas las pestañas en una pieza de trabajo comparten el mismo ancho

**Eliminar una pestaña:**
1. Haz clic en la manija de la pestaña para seleccionarla
2. Presiona Eliminar o usa el menú contextual
3. O limpia todas las pestañas y comienza de nuevo

### Habilitar/Deshabilitar Pestañas

Alterna pestañas on/off sin eliminarlas:

- **Panel de propiedades de pieza de trabajo:** Checkbox "Habilitar Pestañas"
- **Barra de herramientas:** Icono de alternar visibilidad de pestañas

**Cuando está deshabilitado:**
- Las pestañas no se generan en el G-code
- Las manijas de pestañas están ocultas en el lienzo
- La trayectoria corta completamente a través

**Caso de uso:** Deshabilita temporalmente las pestañas para probar el corte, luego rehabilita para producción.

---

## Remover Pestañas Después del Corte

**Herramientas:**
- Cuchilla de manualidades o cutter
- Alicates de corte flush
- Cincel (para madera)
- Sierra fina para materiales gruesos

**Técnica:**

1. **Marca la pestaña** desde ambos lados si es accesible
2. **Dobla suavemente** la pieza para estresar la pestaña
3. **Corta a través** del material restante
4. **Lija o lima** el remanente de la pestaña al ras con el borde

**Para materiales frágiles (acrílico):**
- Usa pestañas mínimas (se rompen fácilmente)
- Marca profundamente antes de romper
- Soporta la pieza mientras rompes pestañas para evitar grietas

**Para madera:**
- Las pestañas pueden requerir corte (no se rompen limpiamente)
- Usa un cuchillo afilado o cincel
- Corta al ras, luego lija suavemente

---

## Páginas Relacionadas

- [Corte de Contorno](operations/contour) - Operación principal que usa pestañas
- [Flujo de Trabajo Multi-Capa](multi-layer) - Gestionar pestañas a través de múltiples capas
- [Previsualización 3D](../ui/3d-preview) - Visualizar pestañas en previsualización
- [Modo Simulación](simulation-mode) - Previsualizar cortes con huecos de pestaña
