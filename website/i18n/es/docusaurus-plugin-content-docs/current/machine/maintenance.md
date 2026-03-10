# Mantenimiento

La página de Mantenimiento en Configuración de Máquina te ayuda a rastrear el uso de la máquina y programar tareas de mantenimiento.

![Ajustes de Mantenimiento](/screenshots/machine-maintenance.png)

## Seguimiento de Uso

Rayforge rastrea cuánto tiempo ha estado en uso tu máquina. Esta información te ayuda a programar el mantenimiento preventivo en intervalos apropiados.

### Horas Totales

El contador de horas totales rastrea todo el tiempo gastado ejecutando trabajos en la máquina. Este contador acumulativo no puede reiniciarse y proporciona un historial completo del uso de la máquina.

Usa esto para rastrear la edad general de la máquina y planificar intervalos de servicio importantes.

## Contadores de Mantenimiento Personalizados

Puedes crear contadores personalizados para rastrear intervalos de mantenimiento específicos. Cada contador tiene un nombre, rastrea horas y puede configurarse con un umbral de notificación.

### Creando un Contador

1. Haz clic en el botón añadir para crear un nuevo contador
2. Ingresa un nombre descriptivo (ej., "Tubo Láser", "Tensión de Correa", "Limpieza de Espejo")
3. Establece un umbral de notificación en horas si lo deseas

### Características del Contador

- **Nombres personalizados**: Etiqueta contadores para cualquier tarea de mantenimiento
- **Seguimiento de horas**: Acumula automáticamente tiempo durante la ejecución de trabajos
- **Umbrales de notificación**: Recibe recordatorios cuando el mantenimiento es necesario
- **Capacidad de reinicio**: Reinicia contadores después de realizar mantenimiento

### Contadores de Ejemplo

**Tubo Láser**: Rastrea horas del tubo CO2 para planificar reemplazo (típicamente 1000-3000 horas). Establece una notificación a las 2500 horas para planificar con anticipación.

**Tensión de Correa**: Rastrea horas desde la última tensión de correa. Reinicia después de realizar el mantenimiento.

**Limpieza de Espejo**: Rastrea uso desde la última limpieza de espejo. Reinicia después de limpiar.

**Lubricación de Rodamientos**: Rastrea horas para intervalos de mantenimiento de rodamientos.

## Reiniciando Contadores

Después de realizar el mantenimiento, puedes reiniciar el contador relevante:

1. Haz clic en el botón de reinicio junto al contador
2. Confirma el reinicio en el diálogo
3. El contador vuelve a cero

:::tip Programa de Mantenimiento
Intervalos de mantenimiento comunes:
- **Diario**: Limpiar lente, revisar alineación de espejos
- **Semanal**: Limpiar rieles, revisar tensión de correas
- **Mensual**: Lubricar rodamientos, revisar conexiones eléctricas
- **Anual**: Inspección completa, reemplazar piezas desgastadas

Ajusta los intervalos basándote en tus patrones de uso y recomendaciones del fabricante.
:::

## Ver También

- [Ajustes de Láser](laser) - Configuración del cabezal láser
- [Ajustes de Hardware](hardware) - Dimensiones de la máquina
