# Modo de Simulación

![Modo de Simulación](/screenshots/main-simulation.png)

El modo de simulación muestra cómo se ejecutará tu trabajo de láser antes de ejecutarlo en la máquina. Puedes recorrer el código G paso a paso y ver exactamente qué sucederá.

## Activar el Modo de Simulación

- **Teclado**: Pulsa <kbd>F11</kbd>
- **Menú**: Ve a **Ver → Simular Ejecución**
- **Barra de herramientas**: Haz clic en el botón de simulación

## Visualización

### Mapa de Calor de Velocidad

Las operaciones se colorean según la velocidad:

| Velocidad  | Color    |
| ---------- | -------- |
| Más lenta  | Azul     |
| Lenta      | Cian     |
| Media      | Verde    |
| Rápida     | Amarillo |
| Más rápida | Rojo     |

Los colores son relativos al rango de velocidad de tu trabajo - el azul es el mínimo, el rojo es el máximo.

### Transparencia de Potencia

La opacidad de las líneas muestra la potencia del láser:

- **Líneas tenues** = Potencia baja (movimientos de desplazamiento, grabado ligero)
- **Líneas sólidas** = Potencia alta (corte)

## Controles de Reproducción

Usa los controles en la parte inferior del lienzo:

- **Reproducir/Pausar** (<kbd>Espacio</kbd>): Iniciar o detener la reproducción automática
- **Control deslizante de progreso**: Arrastra para desplazarte por el trabajo
- **Teclas de flecha**: Recorrer las instrucciones una por una

La simulación y la vista de código G se mantienen sincronizadas - recorrer la simulación resalta el código G correspondiente, y hacer clic en líneas de código G salta a ese punto en la simulación.

## Editar Durante la Simulación

Puedes editar las piezas de trabajo mientras simulas. Mueve, escala o rota objetos, y la simulación se actualiza automáticamente.

## Temas Relacionados

- **[Vista Previa 3D](../ui/3d-preview)** - Visualización de trayectoria 3D
- **[Cuadrícula de Prueba de Material](operations/material-test-grid)** - Usa la simulación para validar pruebas
