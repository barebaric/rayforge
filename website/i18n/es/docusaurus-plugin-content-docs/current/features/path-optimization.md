# Optimización de Trayectoria

La optimización de trayectoria reordena los segmentos de corte para minimizar la distancia de viaje. El láser se mueve eficientemente entre cortes en lugar de saltar aleatoriamente a través del área de trabajo.

## Cómo Funciona

Sin optimización, las trayectorias se cortan en el orden en que aparecen en tu archivo de diseño. La optimización analiza todos los segmentos de trayectoria y los reorganiza para que el láser viaje la distancia total más corta entre cortes.

**Antes de optimización:** El láser salta de un lado a otro a través del material  
**Después de optimización:** El láser se mueve secuencialmente de corte a corte

## Ajustes

### Habilitar Optimización

Activa o desactiva la optimización de trayectoria. Habilitada por defecto para la mayoría de operaciones.

## Cuándo Usar Optimización

**Habilitar para:**

- Diseños con muchas formas separadas
- Reducir el tiempo total del trabajo
- Minimizar el desgaste del sistema de movimiento
- Diseños anidados complejos

**Deshabilitar para:**

- Diseños donde el orden de corte importa (ej., características interiores antes que exteriores)
- Depuración de problemas de trayectoria
- Cuando necesitas un orden de ejecución predecible y repetible

## Cómo Afecta Tu Trabajo

**Ahorro de tiempo:** Puede reducir el tiempo de trabajo en 20-50% para diseños con muchos cortes separados.

**Eficiencia de movimiento:** Menos movimiento rápido significa menos desgaste en correas, motores y rodamientos.

**Distribución de calor:** Las trayectorias optimizadas pueden concentrar calor en un área. Para materiales sensibles al calor, considera si el orden importa.

:::tip
La optimización se ejecuta automáticamente. Simplemente habilítala y el software maneja el resto.
:::

---

## Temas Relacionados

- [Corte de Contorno](operations/contour) - Operación de corte principal
- [Pestañas de Sujeción](holding-tabs) - Mantener piezas aseguradas durante el corte
