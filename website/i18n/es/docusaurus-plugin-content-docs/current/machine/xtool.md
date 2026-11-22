---
description:
  "Conecta cortadoras láser xTool D1 y D1 Pro y la xTool M2 por Wi-Fi con los controladores xTool
  experimentales de Rayforge."
---

# xTool por Wi-Fi

Rayforge incluye controladores experimentales para máquinas xTool conectadas por Wi-Fi. Usan la API
de red propietaria de xTool en lugar de las interfaces GRBL Telnet o FluidNC / ESP3D, por lo que la
máquina debe configurarse con su propio controlador xTool.

## Máquinas compatibles

| Perfil           | Controlador      | Qué puede hacer                                                                              |
| ---------------- | ---------------- | -------------------------------------------------------------------------------------------- |
| **xTool D1 Pro** | xTool (Wi-Fi)    | Subir e iniciar trabajos, referenciar, mover, pausar, reanudar y cancelar                    |
| **xTool M2**     | xTool M2 (Wi-Fi) | Supervisar la máquina, referenciar, mover y pausar, reanudar o cancelar un trabajo preparado |

Ambos controladores están marcados como **experimentales**. Pruébalos con un trozo de material de
descarte antes de confiar en ellos para producción y notifica los problemas adjuntando un registro
de depuración.

## Configurar una máquina xTool

1. Conecta la máquina a la misma red que tu ordenador y averigua su dirección IP (por ejemplo, en la
   aplicación de xTool o en la lista de dispositivos del router).
2. Añade una máquina y elige el perfil **xTool D1 Pro** o **xTool M2**. El perfil ya configura el
   controlador y las dimensiones de la máquina.
3. Introduce el **Hostname o dirección IP** de la máquina.
4. Deja los puertos en sus valores predeterminados salvo que tu red requiera otra cosa:
   - D1 Pro: puerto HTTP `8080`, puerto WebSocket `8081`
   - M2: puerto `28900`
5. Conecta.

Si Rayforge indica que ninguna máquina xTool respondió, comprueba la dirección IP, que la máquina
esté encendida y que el ordenador y la máquina estén en la misma red.

## D1 y D1 Pro

El controlador **xTool (Wi-Fi)** sube el trabajo a la máquina, lo inicia y sigue su progreso
mediante los eventos de trabajo del WebSocket. Las referencias, los movimientos manuales y
absolutos, la pausa, la reanudación y la cancelación funcionan como de costumbre.

La API de red de la máquina no informa su posición, así que Rayforge deduce la posición del estado
del trabajo. La referencia está disponible para todos los ejes a la vez; la referencia por eje está
desactivada. Este controlador no admite escribir ajustes de firmware, desplazamientos de coordenadas
de trabajo ni sondeo.

## xTool M2

El controlador **xTool M2 (Wi-Fi)** se conecta al servicio TLS del M2 en el puerto `28900`.
Supervisa el estado de la máquina y admite referencias, movimientos manuales y pausar, reanudar o
cancelar un trabajo que ya esté preparado en la máquina.

La API de red del M2 todavía no puede recibir G-code, así que Rayforge no puede enviarle sus propios
trabajos. Prepara el trabajo en xTool Studio, inícialo allí y usa Rayforge para supervisarlo y
controlarlo. No se admiten movimientos Z, cambios de herramienta, control remoto de potencia láser,
ajustes de firmware ni desplazamientos de coordenadas de trabajo.

## Certificados TLS

El controlador del M2 valida el certificado TLS de la máquina. Si la máquina usa un certificado
autofirmado, activa **Permitir certificados TLS no válidos** en los ajustes del controlador. Hazlo
solo en una red de confianza.

## Véase también

- [Ajustes generales](general) - Selección de controlador, hostname y puertos
- [Referencia de firmware](../reference/firmware.md) - Controladores y drivers compatibles
- [Dispositivos remotos](remote) - Conectar con una máquina a través de un daemon raydriver
