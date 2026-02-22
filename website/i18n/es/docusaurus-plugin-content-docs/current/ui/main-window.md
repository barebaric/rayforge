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
- **Operaciones**: Añadir, editar y gestionar operaciones
- **Máquina**: Conectar, desplazar, origen, iniciar/detener trabajos
- **Ayuda**: Documentación, acerca de y soporte

### 2. Barra de Herramientas

Acceso rápido a herramientas usadas frecuentemente:

- **Herramienta de selección**: Seleccionar y mover objetos
- **Herramienta de desplazamiento**: Navegar el lienzo
- **Herramienta de zoom**: Acercar/alejar en áreas específicas
- **Herramienta de medición**: Medir distancias y ángulos
- **Herramientas de alineación**: Alinear y distribuir objetos
- **Menú desplegable WCS**: Seleccionar el Sistema de Coordenadas de Trabajo activo (G53-G59)

El menú desplegable WCS te permite cambiar rápidamente entre sistemas de coordenadas.
Ver [Sistemas de Coordenadas de Trabajo](../general-info/work-coordinate-systems) para
más información.

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

### 6. Panel de Control

El Panel de Control en la parte inferior de la ventana proporciona:

- **Controles de Desplazamiento**: Movimiento y posicionamiento manual de la máquina
- **Estado de la Máquina**: Posición en tiempo real y estado de conexión
- **Vista de Registro**: Comunicación de código G e historial de operaciones
- **Gestión de WCS**: Selección y puesta a cero del sistema de coordenadas de trabajo

Ver [Panel de Control](control-panel) para información detallada.

## Gestión de Ventanas

### Paneles

Muestra/oculta paneles según sea necesario:

- **Panel de Capas**: Ver → Panel de Capas (<kbd>ctrl+l</kbd>)
- **Panel de Propiedades**: Ver → Panel de Propiedades (<kbd>ctrl+i</kbd>)

### Modo Pantalla Completa

Enfócate en tu trabajo con pantalla completa:

- Entrar: <kbd>f11</kbd> o Ver → Pantalla Completa
- Salir: <kbd>f11</kbd> o <kbd>esc</kbd>

## Personalización

Personaliza la interfaz en **Editar → Preferencias**:

- **Tema**: Claro, oscuro o del sistema
- **Unidades**: Milímetros o pulgadas
- **Cuadrícula**: Mostrar/ocultar y configurar espaciado de cuadrícula
- **Reglas**: Mostrar/ocultar reglas en el lienzo
- **Barra de Herramientas**: Personalizar botones visibles

---

**Páginas Relacionadas:**

- [Sistemas de Coordenadas de Trabajo](../general-info/work-coordinate-systems) - WCS
- [Herramientas del Lienzo](canvas-tools) - Herramientas para manipular diseños
- [Panel de Control](control-panel) - Control manual de máquina, estado y registros
- [Vista Previa 3D](3d-preview) - Visualizar trayectorias en 3D
