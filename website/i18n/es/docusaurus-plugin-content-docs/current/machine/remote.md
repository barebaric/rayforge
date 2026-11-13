---
description:
  "Ejecute su máquina desde un pequeño ordenador junto al hardware mientras Rayforge se ejecuta en
  otro lugar. Los trabajos sobreviven a las desconexiones y al reconectar se sincroniza el progreso
  exactamente."
---

# Dispositivos remotos

Ejecute su máquina desde un pequeño ordenador junto al hardware (por ejemplo, una Raspberry Pi en el
taller) mientras Rayforge se ejecuta en su portátil o escritorio. Los trabajos se ejecutan en el
host del dispositivo y continúan aunque Rayforge se desconecte; al reconectar, el progreso se
sincroniza exactamente.

## Cómo funciona

- El **demonio de raydriver** aloja la conexión de la máquina junto al hardware.
- Rayforge conserva la configuración de su máquina y envía los trabajos a través de la LAN.
- Todos los controladores de G-code (GRBL serie, GRBL telnet, GRBL simple, Smoothie, Marlin)
  funcionan local o remotamente: remoto es un ajuste de conexión, no un controlador aparte.

## Configurar el host del dispositivo

1. Instale raydriver en el host del dispositivo: `pip install raydriver` (añada `[daemon]` para el
   descubrimiento en red).
2. Ejecute `raydriver daemon --setup`. Esto escribe `~/.config/raydriver/daemon.json` con un token
   generado e imprime el comando del certificado.
3. Para el acceso LAN, cree un certificado con el comando `openssl` impreso y configure `tls_cert` y
   `tls_key` en la configuración. Un demonio en una dirección que no sea loopback no arranca sin
   token y TLS.
4. Ejecute `raydriver daemon install` para instalar e iniciar el servicio en segundo plano (systemd
   en Linux, launchd en macOS, una tarea de inicio de sesión en Windows), o ejecute
   `raydriver daemon` en primer plano.

## Conectar Rayforge

- El asistente de máquinas descubre demonios en la red y muestra sus dispositivos.
- Como alternativa, active **Dispositivo remoto** en los ajustes de conexión del controlador e
  introduzca el nombre de host, el puerto, el token y el certificado TLS del demonio.
- El sondeo funciona a través de la conexión remota, por lo que el asistente puede identificar
  hardware que no está conectado a su ordenador.

## Trabajos y reconexiones

- Un trabajo enviado se ejecuta hasta completarse en el host del dispositivo aunque Rayforge se
  cierre o se caiga la red.
- Reconecte para ver el progreso exacto. Un cliente a la vez tiene el control; los demás pueden
  observar.

## Cámaras en el host del dispositivo

El vídeo no viaja por la conexión de control. Use las [cámaras de red](camera.md) de Rayforge con un
flujo accesible en el host del dispositivo:

- **Cámara IP**: añada su URL de flujo directamente.
- **Cámara USB en el host del dispositivo**: ejecute un streamer estándar como `ustreamer` y añada
  su URL MJPEG como cámara de flujo HTTP.

El demonio anuncia los puntos de cámara configurados para que el asistente pueda rellenar la URL.

## Notas de seguridad

- Solo LAN (o túnel): sin nube, sin cuentas.
- El token y TLS son obligatorios para cualquier demonio que no sea loopback.
- Mantenga los flujos de cámara solo en LAN o tras un firewall; están separados del canal de control
  TLS.
