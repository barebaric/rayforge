# Ajustes

![Ajustes Generales](/screenshots/app-settings-general.webp)

Personaliza Rayforge para que coincida con tu flujo de trabajo y preferencias. Abre los ajustes vía
**Editar → Ajustes** o presiona <kbd>ctrl+coma</kbd>.

## General

La página General contiene ajustes de toda la aplicación.

### Apariencia

Elige entre el tema **Sistema**, **Claro** u **Oscuro** para coincidir con tu entorno de escritorio
o preferencia personal. También puedes configurar los **Colores de operación** para usar el color
del láser o el color de la capa como distinción visual en el lienzo.

### Unidades

Configura las unidades de pantalla usadas en toda la aplicación. Puedes establecer unidades
separadas para **longitud** (milímetros, pulgadas, etc.), **velocidad** (mm/min, mm/seg,
pulgadas/min, etc.) y **aceleración** (mm/s², etc.).

### Comportamiento

Por defecto, las operaciones se recalculan automáticamente después de cada cambio. Si trabajas en
una máquina más lenta o con documentos muy complejos, puedes deshabilitar **Auto-actualizar
operaciones** y activar el recálculo manualmente vía el botón de la barra de herramientas.

Rayforge puede **Buscar actualizaciones** automáticamente al inicio. Cuando está habilitado, se te
notificará cuando haya una nueva versión disponible.

También puedes configurar el **Comportamiento de inicio** — iniciar con un espacio de trabajo vacío,
reabrir el último proyecto o abrir siempre un archivo de proyecto específico. Ten en cuenta que los
archivos especificados en la línea de comandos siempre anularán estos ajustes.

### Privacidad

Rayforge puede enviar datos de uso anónimos para ayudar a mejorar la aplicación. No se recopila
información personal. Puedes activar o desactivar **Informar uso anónimo** en cualquier momento.
Visita la página de [seguimiento de uso](https://rayforge.org/docs/general-info/usage-tracking) para
obtener más información sobre qué datos se recopilan y cómo se usan.

## Gestos del ratón

La página de gestos del ratón te permite reasignar los gestos de navegación del lienzo 2D y del
lienzo 3D; el editor de bocetos usa los mismos gestos que el lienzo 2D. Cada fila ofrece un menú
desplegable con las combinaciones de botones del ratón disponibles:

- **Lienzo 2D** — _Desplazar la vista_ (por defecto, arrastrar con el botón central; con el botón
  izquierdo el arrastre está reservado para la selección, por lo que no se ofrece).
- **Lienzo 3D** — _Orbitar la cámara_ (arrastrar con el botón central), _Desplazar la cámara_
  (Mayús + botón central) y _Rotar alrededor del eje Z_ (arrastrar con el botón izquierdo).

El zoom (rueda del ratón), el menú contextual (clic derecho) y el restablecimiento de la vista (la
tecla `1` en el lienzo 2D) son fijos y no se pueden cambiar. Un clic simple sobre un botón asignado
sigue realizando su acción fija: cuando el desplazamiento está asignado al botón derecho del ratón,
arrastrar con el botón derecho desplaza la vista y un clic derecho simple sigue abriendo el menú
contextual. Al seleccionar **Sin asignar** se desactiva un gesto, y las combinaciones que otra
acción del mismo lienzo ya usa no se ofrecen. Los addons pueden aportar configuraciones de gestos
adicionales, que aparecen como secciones extra en esta página.

## Otros ajustes

El diálogo de ajustes también incluye páginas para gestionar otras partes de la aplicación. Cada una
tiene su propia documentación:

- [Máquinas](../application-settings/machines.md) — añadir, eliminar y configurar tus cortadores
  láser
- [Materiales](../application-settings/materials.md) — gestionar tus bibliotecas de materiales
- [Recetas](../application-settings/recipes.md) — gestionar las recetas de operaciones guardadas
- [Reglas de color](../application-settings/color-rules.md) — asignar los colores SVG a tipos de
  paso
- [Proveedores de IA](../application-settings/ai-provider.md) — configurar proveedores de IA para
  uso de los addons
- [Addons](../application-settings/addons.md) — instalar, actualizar y eliminar addons de extensión
