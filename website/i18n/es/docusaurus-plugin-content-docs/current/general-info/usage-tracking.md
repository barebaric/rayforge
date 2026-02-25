# Seguimiento de uso

Rayforge incluye un seguimiento de uso anónimo opcional para ayudarnos a comprender cómo se utiliza la aplicación y priorizar el desarrollo futuro. Esta página explica qué rastreamos, cómo funciona y su privacidad.

## Completamente opcional

El seguimiento de uso es **completamente opcional**. Cuando inicie Rayforge por primera vez, se le preguntará si desea participar:

- **Sí**: Se enviarán datos de uso anónimos a nuestro servidor de análisis
- **No**: Nunca se recopilarán ni transmitirán datos

Puede cambiar esta elección en cualquier momento en la configuración general.

## Qué rastreamos

Cuando está habilitado, solo recopilamos datos anónimos de vistas de página, similares a los análisis de sitios web. Esto es lo que podemos ver:

| Datos                      | Ejemplo                   |
| -------------------------- | ------------------------- |
| Resolución de pantalla     | 1920x1080                 |
| Configuración de idioma    | es-ES                     |
| Páginas/diálogos vistos    | /machine-settings/general |
| Tiempo en la página        | 6m 3s                     |

## Lo que vemos

Aquí hay un ejemplo de cómo se ve el panel de análisis:

| Ruta                      | Visitantes | Visitas | Vistas | Tasa de rebote | Duración de visita |
| ------------------------- | ---------- | ------- | ------ | -------------- | ------------------ |
| /                         | 1          | 1       | 5      | 0%             | 27m 35s            |
| /machine-settings/general | 1          | 1       | 5      | 0%             | 27m 27s            |
| /view/3d                  | 1          | 1       | 2      | 0%             | 25m 14s            |
| /camera-alignment-dialog  | 1          | 1       | 2      | 0%             | 6m 3s              |
| /machine-settings/camera  | 1          | 1       | 2      | 0%             | 6m 16s             |
| /settings/general         | 1          | 1       | 2      | 0%             | 16m 36s            |
| /step-settings/rasterizer | 1          | 1       | 2      | 0%             | 11s                |

## Lo que NO rastreamos

Estamos comprometidos con su privacidad:

- **Sin información personal** – Sin nombres, correos electrónicos ni cuentas
- **Sin contenido de archivos** – Sus diseños y proyectos permanecen privados
- **Sin identificadores de máquina** – Sin números de serie ni IDs únicos
- **Sin direcciones IP almacenadas** – Usamos Umami Analytics que no almacena IPs
- **Sin seguimiento entre sitios** – Los datos están aislados solo para Rayforge

## Por qué rastreamos

Los datos de uso nos ayudan a:

- **Identificar funciones populares** – Saber lo que funciona bien
- **Encontrar puntos de dolor** – Ver dónde los usuarios pasan tiempo o se quedan atascados
- **Priorizar el desarrollo** – Enfocarnos en funciones que la gente realmente usa
- **Entender la diversidad** – Saber qué idiomas y tamaños de pantalla admitir

## Cómo funciona

Rayforge usa [Umami](https://umami.is/), una plataforma de análisis de código abierto centrada en la privacidad. El seguimiento:

- Envía pequeñas solicitudes HTTP en segundo plano
- No afecta el rendimiento de la aplicación
- Funciona sin conexión (las solicitudes fallidas se ignoran silenciosamente)
- Usa un User-Agent genérico para evitar la huella digital

## Desactivar el seguimiento

Puede desactivar el seguimiento en cualquier momento:

1. Abra **Configuración** → **General**
2. Desactive **Enviar estadísticas de uso anónimas**

Cuando está desactivado, no se envían absolutamente ningún dato.

## Páginas relacionadas

- **[Configuración de la aplicación](../ui/settings)** – Configurar preferencias de seguimiento
