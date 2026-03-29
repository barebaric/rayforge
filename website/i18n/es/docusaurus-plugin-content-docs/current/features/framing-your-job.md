# Enmarcando su trabajo

Aprenda a utilizar la función de enmarcado para previsualizar los límites de
su trabajo y asegurar una alineación correcta antes de cortar.

## Resumen

El enmarcado le permite previsualizar los límites exactos de su trabajo láser
trazando un contorno con el láser a baja potencia o con el láser apagado. Esto
ayuda a verificar la posición y evitar errores costosos.

## Cuándo usar el enmarcado

- **Configuraciones iniciales**: Verificar la colocación del material
- **Posicionamiento preciso**: Asegurar que el diseño cabe dentro de los
  límites del material
- **Múltiples trabajos**: Confirmar la alineación antes de cada pasada
- **Materiales costosos**: Verificar antes de realizar los cortes

## Cómo enmarcar

### Método 1: Solo contorno

Trazar el límite del trabajo sin encender el láser:

1. **Cargue su diseño** en Rayforge
2. **Coloque el material** en la cama del láser
3. **Haga clic en el botón Enmarcar** en la barra de herramientas
4. **Observe la cabeza láser** trazar el rectángulo delimitador
5. **Verifique la posición** y ajuste el material si es necesario

### Método 2: Vista previa con baja potencia

Algunas máquinas soportan enmarcado a baja potencia con un haz visible:

1. **Active el modo de baja potencia** en la configuración de la máquina
2. **Configure la potencia de enmarcado** (típicamente 1-5 %)
3. **Ejecute la operación de enmarcado**
4. **Observe el contorno** trazado en la superficie del material

:::warning Verifique su máquina
No todos los láseres soportan el enmarcado a baja potencia de forma segura.
Consulte la documentación de su máquina antes de usar esta función.
:::

## Configuración de enmarcado

Configure el comportamiento del enmarcado en los ajustes de la cabeza láser de
su máquina:

- **Velocidad de enmarcado**: Qué tan rápido se mueve la cabeza láser durante
  el enmarcado. Se configura por cabeza láser, por lo que si su máquina tiene
  múltiples láseres puede usar diferentes velocidades para cada uno.
- **Potencia de enmarcado**: Potencia del láser durante el enmarcado (0 para
  apagado, % bajo para trazo visible)
- **Tiempo de pausa en esquinas**: Una breve pausa en cada esquina del
  contorno. Esto le da un momento para ver exactamente dónde cae cada esquina
  — especialmente útil a velocidades de enmarcado más altas.
- **Cantidad de repeticiones**: Número de veces que se traza el contorno.
  Establecer un valor mayor a uno puede hacer que la ruta sea más fácil de
  seguir a simple vista.

## Uso de los resultados del enmarcado

Después de enmarcar, puede:

- **Ajustar la posición del material** si es necesario
- **Enmarcar nuevamente** para verificar la nueva posición
- **Proceder con el trabajo** una vez satisfecho con la colocación

## Consejos para un enmarcado efectivo

- **Marque las esquinas**: Coloque pequeños trozos de cinta en las esquinas
  como referencia
- **Verifique el espacio**: Asegure espacio adecuado alrededor de su diseño
- **Confirme la orientación**: Verifique que el material esté orientado
  correctamente
- **Considere la holgura de corte**: Recuerde que los cortes serán ligeramente
  más anchos que los contornos

## Enmarcado con cámara

Si su máquina tiene soporte de cámara, puede:

1. **Capturar imagen de la cámara** de la colocación del material
2. **Superponer el diseño** en la vista de cámara
3. **Ajustar la posición** virtualmente antes de enmarcar
4. **Enmarcar para confirmar** la alineación física

Consulte [Integración de cámara](../machine/camera) para más detalles.

## Solución de problemas

**El marco no coincide con el diseño**: Verifique el origen del trabajo y la
configuración del sistema de coordenadas

**El láser dispara durante el enmarcado**: Desactive la potencia de enmarcado
o revise la configuración de la máquina

**El marco es demasiado rápido para ver**: Reduzca la velocidad de enmarcado
en la configuración

**La cabeza no alcanza las esquinas**: Verifique que el diseño está dentro del
área de trabajo de la máquina

## Notas de seguridad

- **Nunca deje la máquina desatendida** durante el enmarcado
- **Verifique que el láser esté apagado** si usa enmarcado sin potencia
- **Mantenga las manos alejadas** de la trayectoria de la cabeza láser
- **Esté atento a obstrucciones** que puedan interferir con el movimiento

## Temas relacionados

- [Posicionamiento de pieza](workpiece-positioning) - Guía completa de
  posicionamiento
- [Integración de cámara](../machine/camera)
- [Guía de inicio rápido](../getting-started/quick-start)
