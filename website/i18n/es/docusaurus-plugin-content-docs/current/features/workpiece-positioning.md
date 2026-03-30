# Guía de Posicionamiento de Piezas de Trabajo

Esta guía cubre todos los métodos disponibles en Rayforge para posicionar
con precisión su pieza de trabajo y alinear sus diseños antes de cortar o
grabar.

## Resumen

El posicionamiento preciso de la pieza de trabajo es esencial para:

- **Prevenir desperdicios**: Evitar cortar en la ubicación incorrecta
- **Alineación precisa**: Posicionar diseños en materiales preimpresos
- **Resultados repetibles**: Ejecutar el mismo trabajo múltiples veces de
  manera consistente
- **Trabajos de múltiples piezas**: Alinear múltiples piezas en una sola hoja

Rayforge proporciona varias herramientas complementarias para el posicionamiento:

| Método                      | Propósito                    | Mejor Para                                       |
| --------------------------- | ---------------------------- | ------------------------------------------------ |
| **Modo Enfoque**            | Ver posición del láser       | Alineación visual rápida                         |
| **Enmarcado**               | Previsualizar límites        | Verificar que el diseño cabe en el material      |
| **Cero SCF**                | Establecer origen            | Posicionamiento repetible                        |
| **Superposición de Cámara** | Colocación visual del diseño | Alineación precisa en características existentes |

---

## Modo Enfoque (Puntero Láser)

El modo enfoque enciende el láser a un nivel de potencia bajo, actuando como
un "puntero láser" para ayudarle a ver exactamente dónde está posicionado el
cabezal del láser.

### Activar el Modo Enfoque

1. **Conectar a su máquina**
2. **Hacer clic en el botón Enfoque** en la barra de herramientas (icono de
   láser)
3. El láser se enciende al nivel de potencia de enfoque configurado
4. **Mover el cabezal del láser** para ver la posición del haz en su material
5. **Hacer clic en el botón Enfoque nuevamente** para apagar cuando termine

:::warning Seguridad
Incluso a baja potencia, el láser puede dañar los ojos. Nunca mire
directamente al haz ni lo apunte a superficies reflectantes. Use protección
ocular adecuada.
:::

### Configurar la Potencia de Enfoque

La potencia de enfoque determina qué tan brillante aparece el punto láser:

1. Vaya a **Configuración → Máquina → Láser**
2. Encuentre la configuración **Potencia de Enfoque**
3. Establezca un valor que haga el punto visible sin marcar su material
   - Valores típicos: 1-5% para la mayoría de materiales
   - Establezca en 0 para desactivar la función

:::tip Encontrar la Potencia Correcta
Comience con 1% y aumente gradualmente. El punto debe ser visible pero no
dejar ninguna marca en su material. Los materiales más oscuros pueden
requerir mayor potencia para ver el punto claramente.
:::

### Cuándo Usar el Modo Enfoque

- **Verificaciones rápidas de alineación**: Ver si el láser está
  aproximadamente donde espera
- **Encontrar bordes del material**: Mover a las esquinas para verificar la
  colocación del material
- **Establecer origen SCF**: Posicionar láser en el punto cero deseado antes
  de establecer SCF
- **Verificar posición de inicio**: Comprobar que el referenciado funcionó
  correctamente

---

## Enmarcado

El enmarcado traza el rectángulo delimitador de su trabajo a potencia baja
(o cero), mostrando exactamente dónde se cortará o grabará su diseño.

### Cómo Enmarcar

1. **Cargar y posicionar su diseño** en Rayforge
2. **Hacer clic en Máquina → Enmarcar** o presionar `Ctrl+F`
3. El cabezal del láser traza el cuadro delimitador de su trabajo
4. **Verificar el contorno** que cabe dentro de su material

### Configuración de Enmarcado

Configurar el comportamiento de enmarcado en **Configuración → Máquina →
Láser**:

- **Velocidad de Enmarcado**: Qué tan rápido se mueve el cabezal durante el
  enmarcado (más lento = más fácil de ver)
- **Potencia de Enmarcado**: Potencia del láser durante el enmarcado
  - Establezca en 0 para enmarcado en aire (láser apagado, solo movimiento)
  - Establezca en 1-5% para un rastro visible en el material

:::tip Enmarcado en Aire vs. Baja Potencia

- **Enmarcado en aire (0% potencia)**: Seguro para cualquier material, pero
  solo ve el movimiento del cabezal
- **Enmarcado de baja potencia**: Deja una marca visible débil, útil para
  alineación precisa en materiales oscuros
  :::

### Cuándo Enmarcar

- **Antes de cada trabajo**: Verificación rápida de que el diseño cabe
- **Después de cambios de posición**: Confirmar que la nueva colocación es
  correcta
- **Materiales costosos**: Verificar dos veces antes de comprometerse
- **Trabajos de múltiples piezas**: Verificar que todas las piezas caben en
  el material

Ver [Enmarcar su Trabajo](framing-your-job) para más detalles.

---

## Establecer Cero SCF (Sistema de Coordenadas de Trabajo)

Los Sistemas de Coordenadas de Trabajo (SCF) le permiten definir "puntos
cero" personalizados para sus trabajos. Esto facilita alinear trabajos a la
posición de su material.

### Configuración Rápida de SCF

1. **Mover el cabezal del láser** a la esquina de su material (o punto de
   origen deseado)
2. **Abrir el Panel de Control** (`Ctrl+L`)
3. **Seleccionar un SCF** (G54 es el sistema de coordenadas de trabajo por
   defecto)
4. **Hacer clic en Cero X y Cero Y** para establecer la posición actual como
   origen
5. El punto (0,0) de su diseño ahora se alineará con esta posición

### Entender los Sistemas de Coordenadas

Rayforge usa varios sistemas de coordenadas:

| Sistema     | Descripción                                          |
| ----------- | ---------------------------------------------------- |
| **G53**     | Coordenadas de máquina (fijas, no se pueden cambiar) |
| **G54**     | Sistema de coordenadas de trabajo 1 (por defecto)    |
| **G55-G59** | Sistemas de coordenadas de trabajo adicionales       |

:::tip Múltiples Áreas de Trabajo
Use diferentes ranuras SCF para diferentes posiciones de fijación. Por
ejemplo:

- G54 para el lado izquierdo de su cama
- G55 para el lado derecho
- G56 para un accesorio rotatorio
  :::

### Cuándo Establecer Cero SCF

- **Nueva colocación de material**: Alinear origen a la esquina del material
- **Trabajo con fijaciones**: Establecer origen al punto de referencia de la
  fijación
- **Trabajos repetibles**: Mismo trabajo, diferentes posiciones
- **Tandas de producción**: Posicionamiento consistente a través de múltiples
  piezas

Ver [Sistemas de Coordenadas de Trabajo](../general-info/coordinate-systems)
para documentación completa.

---

## Posicionamiento Basado en Cámara

La superposición de cámara muestra una vista en vivo de su material con su
diseño superpuesto, permitiendo alineación visual precisa.

### Configurar la Cámara

1. **Conectar una cámara USB** sobre su área de trabajo
2. Vaya a **Configuración → Cámara** y agregue su dispositivo de cámara
3. **Activar la cámara** para ver la superposición en su lienzo
4. **Alinear la cámara** usando el procedimiento de alineación (requerido
   para posicionamiento preciso)

### Alineación de Cámara

La alineación de cámara mapea los píxeles de la cámara a coordenadas del
mundo real:

1. Abrir **Cámara → Alinear Cámara**
2. Colocar marcadores de alineación en posiciones conocidas (al menos 4
   puntos)
3. Ingresar las coordenadas X/Y del mundo real para cada punto
4. Hacer clic en **Aplicar** para calcular la transformación

:::tip Precisión de Alineación

- Use puntos distribuidos en toda su área de trabajo
- Mida las coordenadas del mundo cuidadosamente con una regla
- Use posiciones de máquina (mover a coordenadas conocidas) para mayor
  precisión
  :::

### Posicionamiento con Superposición de Cámara

1. **Activar la superposición de cámara** para ver su material
2. **Importar su diseño**
3. **Arrastrar el diseño** para alinear con características visibles en la
   cámara
4. **Ajuste fino** usando las teclas de flecha para colocación perfecta al
   píxel
5. **Enmarcar para verificar** antes de ejecutar el trabajo

### Cuándo Usar Posicionamiento con Cámara

- **Materiales preimpresos**: Alinear cortes a impresiones existentes
- **Materiales irregulares**: Posicionar en piezas no rectangulares
- **Colocación precisa**: Requisitos de precisión sub-milimétrica
- **Diseños complejos**: Múltiples elementos con espaciado específico

Ver [Integración de Cámara](../machine/camera) para documentación completa.

---

## Flujos de Trabajo Recomendados

### Flujo de Trabajo de Posicionamiento Básico

Para trabajos simples en materiales rectangulares:

1. **Colocar material** en la cama del láser
2. **Activar modo enfoque** y mover para verificar posición del material
3. **Establecer cero SCF** en la esquina del material
4. **Posicionar su diseño** en el lienzo
5. **Enmarcar el trabajo** para verificar colocación
6. **Ejecutar el trabajo**

### Flujo de Trabajo de Alineación de Precisión

Para colocación precisa en materiales preimpresos o marcados:

1. **Configurar y alinear cámara** (configuración única)
2. **Colocar material** en la cama del láser
3. **Activar superposición de cámara** para ver el material
4. **Importar y posicionar diseño** visualmente en la imagen de la cámara
5. **Desactivar cámara** y enmarcar para verificar
6. **Ejecutar el trabajo**

### Flujo de Trabajo de Producción

Para ejecutar múltiples trabajos idénticos:

1. **Configurar fijación** en la cama del láser
2. **Establecer cero SCF** alineado a la fijación (ej. G54)
3. **Cargar y configurar** su diseño
4. **Enmarcar para verificar** alineación con la fijación
5. **Ejecutar el trabajo**
6. **Reemplazar material** y repetir (SCF permanece igual)

### Flujo de Trabajo de Múltiples Posiciones

Para ejecutar el mismo trabajo en diferentes ubicaciones:

1. **Configurar múltiples posiciones SCF**:
   - Mover a posición 1, establecer cero G54
   - Mover a posición 2, establecer cero G55
   - Mover a posición 3, establecer cero G56
2. **Cargar su diseño** (mismo diseño para todas las posiciones)
3. **Seleccionar G54**, enmarcar y ejecutar
4. **Seleccionar G55**, enmarcar y ejecutar
5. **Seleccionar G56**, enmarcar y ejecutar

---

## Solución de Problemas

### Punto láser no visible en modo enfoque

- **Aumentar potencia de enfoque** en configuración de láser
- **Materiales oscuros** pueden requerir mayor potencia (5-10%)
- **Verificar conexión del láser** y asegurar que la máquina responde
- **Verificar potencia de enfoque** no esté establecida en 0

### Superposición de cámara desalineada

- **Volver a ejecutar alineación de cámara** con más puntos de referencia
- **Verificar montaje de cámara** - puede haberse movido
- **Verificar coordenadas del mundo** fueron medidas con precisión
- **Ver solución de problemas de cámara** en documentación de Integración de
  Cámara

---

## Temas Relacionados

- [Enmarcar su Trabajo](framing-your-job) - Documentación detallada de
  enmarcado
- [Sistemas de Coordenadas de Trabajo](../general-info/coordinate-systems) -
  Referencia SCF
- [Integración de Cámara](../machine/camera) - Configuración y alineación de
  cámara
- [Panel de Control](../ui/bottom-panel) - Controles de movimiento y gestión
  SCF
- [Guía de Inicio Rápido](../getting-started/quick-start) - Flujo de trabajo
  básico
