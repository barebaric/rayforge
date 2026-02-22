# Sistemas de Coordenadas

Entender cómo Rayforge maneja los sistemas de coordenadas es esencial para posicionar tu trabajo correctamente.

## Sistema de Coordenadas de Trabajo (WCS) vs Coordenadas de Máquina

Rayforge usa dos sistemas de coordenadas principales:

### Sistema de Coordenadas de Trabajo (WCS)

El WCS es el sistema de coordenadas de tu trabajo. Cuando posicionas un diseño en (50, 100) en el lienzo, esas son coordenadas WCS.

- **Origen**: Definido por ti (por defecto es G54)
- **Propósito**: Diseño y posicionamiento de trabajo
- **Múltiples sistemas**: G54-G59 disponibles para diferentes configuraciones

### Coordenadas de Máquina

Las coordenadas de máquina son posiciones absolutas relativas a la posición de origen de la máquina.

- **Origen**: Origen de la máquina (0,0,0) - fijado por hardware
- **Propósito**: Posicionamiento físico en la cama
- **Fijo**: No puede ser cambiado por software

**Relación**: Los desplazamientos WCS definen cómo las coordenadas de tu trabajo se mapean a las coordenadas de máquina. Si el desplazamiento G54 es (100, 50, 0), entonces tu diseño en WCS (0, 0) corta en la posición de máquina (100, 50).

## Configurando Coordenadas en Rayforge

### Estableciendo el Origen WCS

Para posicionar tu trabajo en la máquina:

1. **Lleva la máquina al origen** primero (comando `$H` o botón Home)
2. **Desplaza la cabeza del láser** a tu posición de origen deseada
3. **Establece cero WCS** usando el Panel de Control:
   - Haz clic en "Cero X" para establecer la X actual como origen
   - Haz clic en "Cero Y" para establecer la Y actual como origen
4. Tu trabajo ahora comenzará desde esta posición

### Seleccionando un WCS

Rayforge soporta los sistemas de coordenadas de trabajo G54-G59:

| Sistema | Caso de Uso |
|---------|-------------|
| G54 | Por defecto, área de trabajo principal |
| G55-G59 | Posiciones de fijación adicionales |

Selecciona el WCS activo en el Panel de Control. Cada sistema almacena su propio desplazamiento desde el origen de la máquina.

### Dirección del Eje Y

Algunas máquinas tienen Y aumentando hacia abajo en lugar de hacia arriba. Configura esto en:

**Configuración → Máquina → Hardware → Ejes**

Si tus trabajos salen reflejados verticalmente, alterna el ajuste de dirección del eje Y.

## Problemas Comunes

### Trabajo en Posición Incorrecta

- **Verifica el desplazamiento WCS**: Envía `G10 L20 P1` para ver el desplazamiento G54
- **Verifica el homing**: La máquina debe ser llevada al origen para posicionamiento consistente
- **Revisa la dirección del eje Y**: Puede estar invertida

### Deriva de Coordenadas Entre Trabajos

- **Siempre haz homing antes de trabajos**: Establece una referencia consistente
- **Revisa desplazamientos G92**: Límpialos con el comando `G92.1`

---

## Páginas Relacionadas

- [Sistemas de Coordenadas de Trabajo (WCS)](work-coordinate-systems) - Gestionando WCS en Rayforge
- [Panel de Control](../ui/control-panel) - Controles de desplazamiento y botones WCS
- [Exportando Código G](../files/exporting) - Opciones de posicionamiento de trabajo
