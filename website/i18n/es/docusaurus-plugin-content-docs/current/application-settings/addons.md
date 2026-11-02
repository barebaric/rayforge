# Addons

![Ajustes de Addons](/screenshots/app-settings-addons.webp)

La página de Addons muestra los addons de extensión instalados y te permite gestionarlos.

## Resumen

Los addons son extensiones que añaden funcionalidad a Rayforge. Pueden proporcionar:

- Nuevas operaciones
- Filtros de importación/exportación
- Controladores de máquina
- Mejoras de interfaz de usuario
- Bibliotecas de materiales

## Servicios en línea

La opción **Habilitar servicios externos de addons** controla el acceso al registro en línea, la
instalación remota y los servicios de licencias de Gumroad y Patreon. Cuando está deshabilitada, los
addons integrados y los ya instalados localmente siguen disponibles, pero Rayforge no realiza
solicitudes a servicios externos de addons.

## Addons Instalados

La lista de addons muestra todos los addons instalados con:

- **Nombre**: El nombre para mostrar del addon
- **Versión**: Versión instalada actualmente
- **Descripción**: Breve descripción de la funcionalidad
- **Estado**: Si está habilitado o deshabilitado

## Gestionando Addons

### Instalando Addons

1. Haz clic en el botón **Instalar**
2. Selecciona un archivo de addon (`.zip` o repositorio de addons)
3. Confirma la instalación (los addons de la comunidad requieren confirmar un aviso de confianza)
4. Espera a que la instalación se complete

### Actualizando Addons

Cuando hay actualizaciones disponibles, un indicador aparece en el diálogo del Registro de Addons.
Haz clic en **Actualizar** para instalar las actualizaciones disponibles.

### Eliminando Addons

1. Selecciona el addon a eliminar
2. Haz clic en el botón **Eliminar**
3. Confirma la eliminación

<!-- prettier-ignore-start -->
:::warning
Eliminar un addon puede afectar documentos que usan sus características.
:::
<!-- prettier-ignore-end -->

## Addons de la Comunidad

Los addons que no fueron creados por el equipo de Rayforge se marcan como addons de la comunidad con
un icono dedicado en las listas de addons.

Al instalar un addon de la comunidad, un diálogo de confirmación explica que el addon no lo
proporciona el propio Rayforge y muestra un enlace al repositorio del addon. Abre el enlace en tu
navegador para revisar el código y su autor antes de continuar.

<!-- prettier-ignore-start -->
:::warning
Los addons tienen el mismo acceso a tu ordenador y a tus archivos que el propio Rayforge. Instala
addons de la comunidad solo si confías en sus autores.
:::
<!-- prettier-ignore-end -->

## Fuentes de Addons

Los addons pueden instalarse desde:

- **Repositorios**: Repositorios Git que contienen addons
- **Registro oficial**: Addons curados por el equipo de Rayforge

## Temas Relacionados

- [Máquinas](machines) - Configuración de máquina
- [Materiales](materials) - Bibliotecas de materiales
- [Recetas](recipes) - Recetas de operaciones
