---
description:
  "Gestiona máquinas en Rayforge: agrega, configura, exporta, importa y cambia entre diferentes
  cortadoras y grabadoras láser para tus proyectos."
---

# Máquinas

![Ajustes de Máquinas](/screenshots/app-settings-machines.webp)

La página **Máquinas** muestra una lista de todas las máquinas configuradas. Cada entrada muestra el
nombre de la máquina y tiene botones para editarla o eliminarla. La máquina activa actualmente está
marcada con un icono de verificación.

Abre la página desde la ventana principal con **Máquina → Manage Machines…**, o selecciona
**Máquinas** en Configuración de la Aplicación (<kbd>ctrl+coma</kbd>).

## Añadir una Máquina

La forma más rápida de añadir una máquina es **Máquina → Add Machine…** en la ventana principal, que
abre el asistente de configuración directamente. Desde la página Máquinas, haz clic en el botón
**Add Machine** en la parte inferior de la lista y luego:

1. Selecciona un perfil de dispositivo de la lista para usarlo como plantilla — cada perfil
   preconfigura los ajustes de la máquina y el dialecto de G-code

![Añadir Máquina](/screenshots/app-settings-machines-add.webp)

2. Se abre el [diálogo de ajustes de máquina](../machine/general.md) donde puedes ajustar la
   configuración

Alternativamente:

- Haz clic en **Device Not Listed** para iniciar el
  [Asistente de Configuración](../getting-started/first-time-setup.md), que te guía en la
  configuración de una máquina paso a paso
- Haz clic en **Import from File…** para añadir una máquina desde un perfil exportado previamente o
  desde un perfil de dispositivo LightBurn (.lbdev). Los perfiles de LightBurn incluyen calibración
  de cámara y configuración del láser que se aplican a la nueva máquina.

## Editar una Máquina

Haz clic en el icono de edición junto a una máquina para abrir el
[diálogo de ajustes de máquina](../machine/general.md).

## Notas de la máquina

La categoría **Notes**, justo después de **General**, reúne las instrucciones del perfil del
dispositivo y tus propias notas para esa máquina. Las notas del dispositivo son de solo lectura;
**My Notes** se pueden editar con Markdown. Se admiten encabezados, párrafos, negrita, cursiva,
listas, enlaces, código en línea, bloques de código y secciones desplegables con `:::details`.

Cuando un perfil proporciona instrucciones, también aparecen en la página final **Review & Name**
del asistente de configuración y siguen disponibles en **Machine Settings → Notes** después de crear
la máquina. Las notas personales se guardan por máquina y no se reemplazan al revisar
actualizaciones del perfil.

## Cambiar la Máquina Activa

El submenú **Máquina → Select Machine** lista todas las máquinas configuradas; la máquina activa
tiene una marca de verificación. Selecciona otra entrada para cambiar a ella. También puedes usar el
menú desplegable de máquinas en la cabecera de la ventana principal. La selección se recuerda entre
sesiones.

## Eliminar una Máquina

1. Haz clic en el icono de eliminación junto a la máquina
2. Confirma la eliminación

<!-- prettier-ignore-start -->
:::warning
Eliminar una máquina no se puede deshacer. Exporta el perfil primero si deseas conservar
la configuración.
:::
<!-- prettier-ignore-end -->
