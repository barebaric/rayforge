---
description: "Gestiona máquinas en Rayforge: agrega, configura, exporta, importa y cambia entre diferentes cortadoras y grabadoras láser para tus proyectos."
---

# Máquinas

![Ajustes de Máquinas](/screenshots/application-machines.png)

La página de Máquinas en Configuración de la Aplicación muestra una lista de
todas las máquinas configuradas. Cada entrada muestra el nombre de la máquina
y tiene botones para editarla o eliminarla. La máquina activa actualmente está
marcada con un icono de verificación.

## Añadir una Máquina

1. Haz clic en el botón **Add Machine** en la parte inferior de la lista
2. Selecciona un perfil de dispositivo de la lista para usarlo como
   plantilla — cada perfil preconfigura los ajustes de la máquina y el
   dialecto de G-code

![Añadir Máquina](/screenshots/add-machine-dialog.png)

3. Se abre el [diálogo de ajustes de máquina](../machine/general) donde
   puedes ajustar la configuración

Alternativamente, haz clic en **Import from File...** en el selector de
perfiles para añadir una máquina desde un perfil exportado previamente.

## Editar una Máquina

Haz clic en el icono de edición junto a una máquina para abrir el
[diálogo de ajustes de máquina](../machine/general).

## Cambiar la Máquina Activa

Usa el menú desplegable de máquinas en la cabecera de la ventana principal
para cambiar entre las máquinas configuradas. La selección se recuerda entre
sesiones.

## Eliminar una Máquina

1. Haz clic en el icono de eliminación junto a la máquina
2. Confirma la eliminación

:::warning
Eliminar una máquina no se puede deshacer. Exporta el perfil primero si
deseas conservar la configuración.
:::
