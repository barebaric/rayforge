# Enmarcando Tu Trabajo

Aprende a usar la función de enmarcado para previsualizar los límites de tu trabajo y asegurar una alineación correcta antes de cortar.

## Resumen

El enmarcado te permite previsualizar los límites exactos de tu trabajo láser trazando un contorno con el láser a baja potencia o con el láser apagado. Esto ayuda a verificar el posicionamiento y prevenir errores costosos.

## Cuándo Usar Enmarcado

- **Configuraciones por primera vez**: Verificar colocación del material
- **Posicionamiento preciso**: Asegurar que el diseño cabe dentro de los límites del material
- **Múltiples trabajos**: Confirmar alineación antes de cada ejecución
- **Materiales costosos**: Verificar doblemente antes de comprometerse a cortes

## Cómo Enmarcar

### Método 1: Solo Contorno

Trazar el límite del trabajo sin encender el láser:

1. **Carga tu diseño** en Rayforge
2. **Coloca el material** en la cama láser
3. **Haz clic en el botón Enmarcar** en la barra de herramientas
4. **Observa el cabezal láser** trazar el rectángulo del límite
5. **Verifica el posicionamiento** y ajusta el material si es necesario

### Método 2: Previsualización de Baja Potencia

Algunas máquinas soportan enmarcado de baja potencia con un haz visible:

1. **Habilita el modo de baja potencia** en ajustes de máquina
2. **Configura potencia de enmarcado** (típicamente 1-5%)
3. **Ejecuta la operación de enmarcado**
4. **Observa el contorno** trazado en la superficie del material

:::warning Verifica Tu Máquina
No todos los láseres soportan enmarcado de baja potencia de manera segura. Consulta la documentación de tu máquina antes de usar esta función.
:::


## Ajustes de Enmarcado

Configura el comportamiento de enmarcado en Ajustes → Máquina:

- **Velocidad de enmarcado**: Qué tan rápido se mueve el cabezal láser durante el enmarcado
- **Potencia de enmarcado**: Potencia del láser durante el enmarcado (0 para apagado, bajo % para trazo visible)
- **Pausa en esquinas**: Breve pausa en cada esquina para visibilidad
- **Conteo de repeticiones**: Número de veces que se traza el contorno

## Usando Resultados del Enmarcado

Después de enmarcar, puedes:

- **Ajustar posición del material** si es necesario
- **Volver a enmarcar** para verificar la nueva posición
- **Proceder con el trabajo** una vez satisfecho con la colocación

## Consejos para Enmarcado Efectivo

- **Marca esquinas**: Coloca pequeños trozos de cinta en las esquinas como referencia
- **Verifica espacio**: Asegura espacio adecuado alrededor de tu diseño
- **Verifica orientación**: Confirma que el material está orientado correctamente
- **Considera el kerf**: Recuerda que los cortes serán ligeramente más anchos que los contornos

## Enmarcado con Cámara

Si tu máquina tiene soporte de cámara, puedes:

1. **Capturar imagen de cámara** de la colocación del material
2. **Superponer diseño** en la vista de cámara
3. **Ajustar posición** virtualmente antes de enmarcar
4. **Enmarcar para confirmar** alineación física

Ver [Integración de Cámara](../machine/camera) para detalles.

## Solución de Problemas

**El enmarcado no coincide con el diseño**: Verifica el origen del trabajo y ajustes del sistema de coordenadas

**El láser dispara durante el enmarcado**: Deshabilita la potencia de enmarcado o verifica los ajustes de máquina

**El enmarcado es muy rápido para ver**: Reduce la velocidad de enmarcado en ajustes

**El cabezal no alcanza las esquinas**: Verifica que el diseño esté dentro del área de trabajo de la máquina

## Notas de Seguridad

- **Nunca dejes la máquina sin supervisión** durante el enmarcado
- **Verifica que el láser esté apagado** si usas enmarcado de potencia cero
- **Mantén las manos alejadas** de la ruta del cabezal láser
- **Observa obstrucciones** que podrían interferir con el movimiento

## Temas Relacionados

- [Integración de Cámara](../machine/camera)
- [Guía de Inicio Rápido](../getting-started/quick-start)
