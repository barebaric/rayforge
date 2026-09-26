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

La página de gestos del ratón te permite reasignar los gestos de navegación del lienzo 2D, del
lienzo 3D y del editor de bocetos. Cada entrada muestra la combinación de botones del ratón asignada
actualmente. Haz clic en una entrada para capturar una nueva asignación: presiona el botón del ratón
deseado (con o sin teclas modificadoras) o usa la rueda del ratón, y la asignación se aplica de
inmediato.

- **Desplazar la vista** — mantén presionado el botón del ratón asignado y mueve para desplazar.
- **Zoom de la vista** — usa la rueda del ratón para hacer zoom.
- **Órbita / rotación (lienzo 3D)** — arrastra para orbitar alrededor de la escena o rotar alrededor
  del eje Z.
- **Abrir el menú contextual** — el menú contextual del lienzo 2D o el menú de herramientas del
  editor de bocetos.
- **Restablecer la vista** — ajusta la vista de nuevo. Sin asignar por defecto.

Una asignación se puede eliminar con **Desasignar** o restaurar con **Restablecer los valores
predeterminados**. Se rechaza la asignación de un gesto que ya se usa para otra acción en la misma
vista, de modo que una combinación de botones del ratón nunca active dos acciones a la vez. Los
addons pueden aportar configuraciones de gestos adicionales, que aparecen como secciones extra en
esta página.

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
