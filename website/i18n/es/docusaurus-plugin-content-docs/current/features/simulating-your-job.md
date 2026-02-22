# Simulando Tu Trabajo

![Captura de Modo Simulación](/screenshots/main-simulation.png)

Aprende a usar el modo de simulación de Rayforge para previsualizar tu trabajo láser, identificar posibles problemas y estimar el tiempo de completado antes de ejecutar en hardware real.

## Resumen

El modo de simulación te permite visualizar la ejecución de tu trabajo láser sin ejecutar realmente la máquina. Esto ayuda a detectar errores, optimizar ajustes y planificar tu flujo de trabajo.

## Beneficios de la Simulación

- **Previsualizar ejecución del trabajo**: Ver exactamente cómo se moverá el láser
- **Estimar tiempo**: Obtener estimaciones precisas de duración del trabajo
- **Identificar problemas**: Detectar superposiciones, huecos o comportamiento inesperado
- **Optimizar orden de trayectoria**: Visualizar secuencia de corte
- **Aprender G-code**: Entender cómo las operaciones se traducen a comandos de máquina

## Iniciar una Simulación

1. **Carga o crea tu diseño** en Rayforge
2. **Configura operaciones** con los ajustes deseados
3. **Haz clic en el botón Simular** en la barra de herramientas (o usa el atajo de teclado)
4. **Observa la simulación** reproducir tu trabajo

## Controles de Simulación

### Controles de Reproducción

- **Reproducir/Pausar**: Iniciar o pausar la simulación
- **Paso Adelante/Atrás**: Mover a través del trabajo un comando a la vez
- **Control de Velocidad**: Ajustar velocidad de reproducción (0.5x a 10x)
- **Saltar a Posición**: Ir a un porcentaje específico del trabajo
- **Reiniciar**: Comenzar simulación desde el inicio

### Opciones de Visualización

- **Mostrar trayectoria de herramienta**: Mostrar la ruta que seguirá el cabezal láser
- **Mostrar movimientos de viaje**: Visualizar movimientos de posicionamiento rápido
- **Mostrar potencia del láser**: Colorear trayectorias por nivel de potencia
- **Modo mapa de calor**: Visualizar tiempo de permanencia y densidad de potencia

### Pantalla de Información

Durante la simulación, monitorea:

- **Posición actual**: Coordenadas X, Y del cabezal láser
- **Progreso del trabajo**: Porcentaje completado
- **Tiempo estimado restante**: Basado en progreso actual
- **Operación actual**: Qué operación se está ejecutando
- **Potencia y velocidad**: Parámetros actuales del láser

## Interpretando Resultados de Simulación

### Qué Buscar

- **Eficiencia de trayectoria**: ¿Hay movimientos de viaje innecesarios?
- **Cortes superpuestos**: Doble corte no intencional de trayectorias
- **Orden de operaciones**: ¿La secuencia tiene sentido?
- **Distribución de potencia**: ¿La potencia se aplica consistentemente?
- **Movimientos inesperados**: Cualquier patrón de movimiento brusco o extraño

### Visualización de Mapa de Calor

El mapa de calor muestra exposición acumulada del láser:

- **Colores fríos (azul/verde)**: Baja exposición
- **Colores cálidos (amarillo/naranja)**: Exposición moderada
- **Colores calientes (rojo)**: Alta exposición o tiempo de permanencia

Usa esto para identificar:

- **Puntos calientes**: Áreas que pueden sobre-quemarse
- **Huecos**: Áreas que pueden estar sub-expuestas
- **Problemas de superposición**: Doble exposición no intencional

Ver [Modo Simulación](../features/simulation-mode) para información detallada.

## Usando Simulación para Optimización

### Optimizar Orden de Corte

Si la simulación revela orden de trayectoria ineficiente:

1. **Habilita optimización de trayectoria** en ajustes de operación
2. **Elige método de optimización** (vecino más cercano, TSP)
3. **Vuelve a simular** para verificar mejora

### Ajustar Tiempos

La simulación proporciona estimaciones de tiempo precisas:

- **Tiempos de trabajo largos**: Considera optimizar trayectorias o aumentar velocidad
- **Tiempos muy cortos**: Verifica que los ajustes son correctos para el material
- **Duración inesperada**: Busca operaciones ocultas o duplicados

### Verificar Trabajos Multi-Capa

Para proyectos complejos multi-capa:

1. **Simula cada capa** independientemente
2. **Verifica orden de operaciones** a través de capas
3. **Busca conflictos** entre capas
4. **Estima tiempo total** para trabajo completo

## Simulación vs. Ejecución Real

### Diferencias a Notar

La simulación es altamente precisa pero:

- **No considera**: Imperfecciones mecánicas, backlash, vibración
- **Puede diferir ligeramente**: Aceleración/desaceleración real vs. simulada
- **No muestra**: Interacción con material, humo, vapores
- **Estimaciones de tiempo**: Generalmente precisas dentro de 5-10%

### Cuándo Volver a Simular

- **Después de cambiar ajustes**: Potencia, velocidad o parámetros de operación
- **Después de editar diseño**: Cualquier cambio de diseño
- **Antes de materiales costosos**: Verifica doblemente antes de comprometerse
- **Al solucionar problemas**: Verifica correcciones a problemas identificados

## Consejos para Simulación Efectiva

- **Siempre simula** antes de ejecutar trabajos importantes
- **Usa reproducción más lenta** para detectar problemas sutiles
- **Habilita mapa de calor** para trabajos de grabado
- **Compara múltiples ajustes** simulando variaciones
- **Documenta resultados**: Captura de pantalla o anota problemas encontrados

## Solución de Problemas de Simulación

**La simulación no inicia**: Verifica que las operaciones estén correctamente configuradas

**La simulación corre muy rápido**: Ajusta velocidad de reproducción a ajuste más lento

**No puedes ver detalles**: Acércate a áreas específicas de interés

**La estimación de tiempo parece incorrecta**: Verifica que el perfil de máquina tenga velocidades máximas correctas

## Temas Relacionados

- [Función Modo Simulación](../features/simulation-mode)
- [Flujo de Trabajo Multi-Capa](../features/multi-layer)
