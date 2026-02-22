# Máquinas

![Ajustes de Máquinas](/screenshots/application-machines.png)

La página de Máquinas en Configuración de la Aplicación te permite gestionar perfiles de máquina. Cada perfil contiene toda la configuración para una máquina láser específica.

## Perfiles de Máquina

Los perfiles de máquina almacenan la configuración completa para un cortador o grabador láser, incluyendo:

- **Ajustes generales**: Nombre, velocidades, aceleración
- **Ajustes de hardware**: Dimensiones del área de trabajo, configuración de ejes
- **Ajustes de láser**: Rango de potencia, frecuencia PWM
- **Ajustes de dispositivo**: Puerto serie, velocidad de transmisión, tipo de firmware
- **Ajustes de código G**: Opciones de dialecto de código G personalizado
- **Ajustes de cámara**: Calibración y alineación de cámara

## Gestionando Máquinas

### Añadiendo una Nueva Máquina

1. Haz clic en el botón **Añadir Nueva Máquina**
2. Ingresa un nombre descriptivo para tu máquina
3. Configura los ajustes de la máquina (ver
   [Configuración de Máquina](../machine/general) para más detalles)
4. Haz clic en **Guardar** para crear el perfil

### Cambiando Entre Máquinas

Usa el menú desplegable de selección de máquina en la ventana principal para cambiar entre
máquinas configuradas. Todos los ajustes, incluyendo la máquina seleccionada, se
recuerdan entre sesiones.

### Duplicando una Máquina

Para crear un perfil de máquina similar:

1. Selecciona la máquina a duplicar
2. Haz clic en el botón **Duplicar**
3. Renombra la nueva máquina y ajusta los ajustes según sea necesario

### Eliminando una Máquina

1. Selecciona la máquina a eliminar
2. Haz clic en el botón **Eliminar**
3. Confirma la eliminación

:::warning
Eliminar un perfil de máquina no se puede deshacer. Asegúrate de haber
anotado cualquier ajuste importante antes de eliminar.
:::

## Temas Relacionados

- [Configuración de Máquina](../machine/general) - Configuración detallada de máquina
- [Configuración Inicial](../getting-started/first-time-setup) - Guía de
  configuración inicial
