# Guía de Inicio Rápido

Ahora que Rayforge está instalado y tu máquina está configurada, ¡ejecutemos tu primer trabajo láser! Esta guía te guiará en la importación de un diseño, la configuración de operaciones y el envío de código G a tu máquina.

## Paso 1: Importar un Diseño

Rayforge soporta varios formatos de archivo incluyendo SVG, DXF, PDF e imágenes rasterizadas (JPEG, PNG, BMP).

1. **Haz clic** en **Archivo → Abrir** o presiona <kbd>ctrl+o</kbd>
2. Navega hasta tu archivo de diseño y selecciónalo
3. El diseño aparecerá en el lienzo

![Lienzo con diseño importado](/screenshots/main-standard.png)

:::tip ¿No tienes un diseño todavía?
Puedes crear formas simples usando las herramientas del lienzo o descargar archivos SVG gratuitos de sitios como [Flaticon](https://www.flaticon.com/) o [SVG Repo](https://www.svgrepo.com/).
:::


## Paso 2: Posicionar Tu Diseño

Usa las herramientas del lienzo para posicionar y ajustar tu diseño:

- **Desplazar**: Clic central y arrastrar, o mantén <kbd>espacio</kbd> y arrastra
- **Zoom**: Rueda del ratón, o <kbd>ctrl+"+"</kbd> / <kbd>ctrl+"-"</kbd>
- **Mover**: Haz clic y arrastra tu diseño
- **Rotar**: Selecciona el diseño y usa las manijas de rotación
- **Escalar**: Selecciona el diseño y arrastra las manijas de las esquinas

## Paso 3: Asignar una Operación

Las operaciones definen cómo Rayforge procesará tu diseño. Las operaciones comunes incluyen:

- **Contorno**: Corta a lo largo del contorno de las formas
- **Grabado Rasterizada**: Rellena formas con líneas de ida y vuelta (para grabar)
- **Grabado de Profundidad**: Crea efectos de profundidad 3D a partir de imágenes

### Añadir una Operación

1. Selecciona tu diseño en el lienzo
2. Haz clic en **Operaciones → Añadir Operación** o presiona <kbd>ctrl+shift+a</kbd>
3. Elige el tipo de operación (ej., "Contorno" para cortar)
4. Configura los ajustes de la operación:
   - **Potencia**: Porcentaje de potencia del láser (¡comienza bajo y prueba!)
   - **Velocidad**: Velocidad de movimiento en mm/min
   - **Pasadas**: Número de veces que se repite la operación (útil para cortar materiales gruesos)

![Ajustes de Operación](/screenshots/step-settings-contour-general.png)

:::warning Comienza con Potencia Baja
Cuando trabajes con materiales nuevos, siempre comienza con ajustes de potencia bajos y ejecuta cortes de prueba. Aumenta gradualmente la potencia hasta lograr el resultado deseado. Usa la función [Cuadrícula de Prueba de Materiales](../features/operations/material-test-grid) para encontrar sistemáticamente los ajustes óptimos.
:::


## Paso 4: Vista Previa

Antes de enviar a tu máquina, previsualiza la trayectoria en 3D:

1. Haz clic en **Ver → Vista Previa 3D** o presiona <kbd>ctrl+3</kbd>
2. La ventana de vista previa 3D muestra la trayectoria completa
3. Usa tu ratón para rotar y hacer zoom en la vista previa
4. Verifica que la trayectoria se vea correcta

![Vista Previa 3D](/screenshots/main-3d.png)

:::tip Detecta Errores Temprano
La vista previa 3D te ayuda a detectar problemas como:

- Trayectorias faltantes
- Orden incorrecto
- Operaciones aplicadas a objetos equivocados
- Trayectorias que exceden tu área de trabajo
:::


## Paso 5: Enviar a la Máquina

:::danger La Seguridad Primero
- Asegúrate de que el área de trabajo esté despejada
- Nunca dejes la máquina desatendida durante la operación
- Ten equipo de seguridad contra incendios cerca
- Usa protección ocular apropiada
:::


### Preparando Tu Material

1. Coloca tu material en la cama láser
2. Enfoca el láser según las instrucciones de tu máquina
3. Si usas la cámara, alinea tu diseño usando la [superposición de cámara](../machine/camera)

### Iniciando el Trabajo

1. **Posiciona el láser**: Usa los controles de desplazamiento para mover el láser a la posición inicial
   - Haz clic en **Ver → Panel de Control** o presiona <kbd>ctrl+l</kbd>
   - Usa los botones de flecha o las flechas del teclado para mover el láser
   - Presiona <kbd>inicio</kbd> para llevar la máquina al origen

2. **Enmarca el diseño**: Ejecuta la función de enmarcado para verificar la ubicación
   - Haz clic en **Máquina → Enmarcar** o presiona <kbd>ctrl+f</kbd>
   - El láser trazará el cuadro delimitador de tu diseño a potencia baja/nula
   - Verifica que quede dentro de tu material

3. **Inicia el trabajo**: Haz clic en **Máquina → Iniciar Trabajo** o presiona <kbd>ctrl+r</kbd>
4. Monitorea el progreso en la barra de estado

### Durante el Trabajo

- La sección derecha de la barra de estado muestra el progreso actual y la estimación del tiempo total de ejecución
- Puedes pausar el trabajo con <kbd>ctrl+p</kbd> o haciendo clic en el botón Pausar
- Presiona <kbd>esc</kbd> o haz clic en Detener para cancelar el trabajo (parada de emergencia)

## Paso 6: Finalizando

Una vez que el trabajo se completa:

1. Espera a que el extractor despeje los humos
2. Retira cuidadosamente tu pieza terminada
3. Limpia la cama láser si es necesario

:::success ¡Felicitaciones!
¡Has completado tu primer trabajo con Rayforge! Ahora puedes explorar funciones más avanzadas.
:::


## Siguientes Pasos

Ahora que has completado tu primer trabajo, explora estas funciones:

- **[Operaciones Multi-Capa](../features/multi-layer)**: Asigna diferentes operaciones a las capas
- **[Pestañas de Sujeción](../features/holding-tabs)**: Mantén las piezas cortadas en su lugar durante el corte
- **[Integración de Cámara](../machine/camera)**: Usa una cámara para alineación precisa
- **[Hooks y Macros](../machine/hooks-macros)**: Automatiza tareas repetitivas

## Consejos para el Éxito

1. **Guarda tu trabajo**: Usa <kbd>ctrl+s</kbd> para guardar tu proyecto frecuentemente
2. **Cortes de prueba**: Siempre ejecuta un corte de prueba en material de desecho primero
3. **Base de datos de materiales**: Mantén notas de los ajustes de potencia/velocidad exitosos para diferentes materiales
4. **Mantenimiento**: Mantén la lente del láser limpia y revisa la tensión de las correas regularmente
5. **Asistencia de aire**: Si tu máquina tiene asistencia de aire, úsala para prevenir carbonización y mejorar la calidad del corte

---

**¿Necesitas Ayuda?** Revisa la sección de [Solución de Problemas](../troubleshooting/connection) o visita la página de [GitHub Issues](https://github.com/barebaric/rayforge/issues).
