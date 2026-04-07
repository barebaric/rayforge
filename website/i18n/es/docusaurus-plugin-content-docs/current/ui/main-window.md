# Ventana Principal

La ventana principal de Rayforge es tu espacio de trabajo principal para crear y gestionar
trabajos láser.

## Diseño de la Ventana

![Ventana Principal](/screenshots/main-standard.png)

### 1. Barra de Menú

Accede a todas las funciones de Rayforge a través de menús organizados:

- **Archivo**: Abrir, guardar, importar, exportar y archivos recientes
- **Editar**: Deshacer, rehacer, copiar, pegar, preferencias
- **Ver**: Zoom, cuadrícula, reglas, paneles y modos de vista
- **Objeto**: Añadir, editar y gestionar operaciones
- **Máquina**: Conectar, desplazar, origen, iniciar/detener trabajos
- **Ayuda**: Acerca de, Donar, Guardar Registro de Depuración

### 2. Barra de Herramientas

Acceso rápido a controles usados frecuentemente:

- **Menú desplegable de máquina**: Selecciona tu máquina, ve el estado de conexión y
  el tiempo estimado durante los trabajos
- **Menú desplegable WCS**: Seleccionar el Sistema de Coordenadas de Trabajo activo (G53-G59)
- **Alternar simulación**: Habilitar/deshabilitar el modo de simulación de trabajo
- **Enfocar láser**: Alternar el modo de enfoque del láser
- **Controles de trabajo**: Botones Home, Enmarcar, Enviar, Pausar y Cancelar

El menú desplegable de máquina muestra el estado de conexión de tu máquina y el estado
actual (ej. Idle, Run) directamente en la barra de herramientas. Durante la ejecución del
trabajo, también muestra el tiempo restante estimado.

El menú desplegable WCS te permite cambiar rápidamente entre sistemas de coordenadas.
Ver [Sistemas de Coordenadas de Trabajo](../general-info/coordinate-systems) para
más información.

Los controles de visibilidad para piezas de trabajo, pestañas, transmisión de cámara,
movimientos de desplazamiento y otros elementos se han movido a botones superpuestos
en el lienzo mismo, para que siempre estén a mano mientras trabajas.

### 3. Lienzo

El espacio de trabajo principal donde:

- Importas y organizas diseños
- Previsualizas trayectorias
- Posicionas objetos relativos al origen de la máquina
- Pruebas límites de enmarcado

**Controles del Lienzo:**

- **Desplazar**: Arrastrar clic central o <kbd>espacio</kbd> + arrastrar
- **Zoom**: Rueda del ratón o <kbd>ctrl+"+"</kbd> / <kbd>ctrl+"-"</kbd>
- **Restablecer Vista**: <kbd>ctrl+0</kbd> o Ver → Restablecer Zoom

### 4. Panel de Capas

Gestiona operaciones y asignaciones de capa:

- Ver todas las operaciones en tu proyecto
- Asignar operaciones a elementos de diseño
- Reordenar la ejecución de operaciones
- Habilitar/deshabilitar operaciones individuales
- Configurar parámetros de operación

### 5. Panel de Propiedades

Configura ajustes para objetos u operaciones seleccionados:

- Tipo de operación (Contorno, Rasterizado, etc.)
- Ajustes de potencia y velocidad
- Número de pasadas
- Opciones avanzadas (overscan, kerf, pestañas)

### 6. Panel Inferior

El Panel Inferior proporciona pestañas para la Consola, el Visor de Código G y los
activos de tu documento (material y bosquejos). Los controles de desplazamiento y la
gestión de WCS siempre están visibles en el lado derecho. El tiempo estimado del trabajo
se muestra en el encabezado de la lista de capas sobre el panel de capas.

Ver [Panel Inferior](bottom-panel) para información detallada.

## Gestión de Ventanas

### Paneles

Muestra/oculta paneles según sea necesario:

- **Panel Inferior**: Ver → Panel Inferior (<kbd>ctrl+l</kbd>)

### Modo Pantalla Completa

Enfócate en tu trabajo con pantalla completa:

- Entrar: <kbd>f11</kbd> o Ver → Pantalla Completa
- Salir: <kbd>f11</kbd> o <kbd>esc</kbd>

## Personalización

Personaliza la interfaz en **Editar → Ajustes**:

- **Tema**: Claro, oscuro o del sistema
- **Unidades**: Milímetros o pulgadas
- **Cuadrícula**: Mostrar/ocultar y configurar espaciado de cuadrícula
- **Reglas**: Mostrar/ocultar reglas en el lienzo

---

**Páginas Relacionadas:**

- [Sistemas de Coordenadas de Trabajo](../general-info/coordinate-systems) - WCS
- [Herramientas del Lienzo](canvas-tools) - Herramientas para manipular diseños
- [Panel Inferior](bottom-panel) - Control manual de máquina, estado y registros
- [Vista Previa 3D](3d-preview) - Visualizar trayectorias en 3D
