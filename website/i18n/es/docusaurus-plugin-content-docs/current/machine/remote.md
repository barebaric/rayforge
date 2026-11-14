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
- Rayforge se conecta a un _dispositivo en el demonio_ — no necesita saber qué protocolo habla el
  dispositivo. El demonio lo resuelve e indica a Rayforge qué puede hacer el dispositivo.
- Las máquinas remotas usan un único controlador **Remote machine (raydriver)**. Los controladores
  específicos de protocolo (GRBL, Smoothie, Marlin) son para hardware conectado directamente a su
  ordenador.

## Configurar el host del dispositivo

1. Instale raydriver en el host del dispositivo: `pip install raydriver`. El descubrimiento en red
   está incluido.
2. Ejecute `raydriver daemon add /dev/ttyUSB0` (use el puerto serie de su máquina; `raydriver scan`
   los lista). Esto crea la configuración con un **token** generado y registra el dispositivo. Añada
   `--protocol`/`--baudrate` para firmware no predeterminado.
3. Ejecute `raydriver daemon install` para instalar e iniciar el servicio en segundo plano (systemd
   en Linux, launchd en macOS, una tarea de inicio de sesión en Windows), o ejecute
   `raydriver daemon` en primer plano.
4. Para el acceso LAN, cree un certificado con el comando `openssl` de una línea que imprime
   `raydriver daemon --setup` y establezca `tls_cert` y `tls_key` en la configuración. Un demonio en
   una dirección que no sea loopback no arranca sin token y TLS.

Consulte la
[guía del demonio de raydriver](https://github.com/barebaric/raydriver/blob/main/docs/daemon.md)
para la referencia completa.

### Obtener el token

El token es el secreto compartido que Rayforge usa para conectarse. Se genera al añadir el primer
dispositivo y se imprime una vez en ese momento. Para volver a imprimirlo, ejecute esto **en el host
del dispositivo**:

```bash
raydriver daemon status
```

Copie el valor de `token:` — lo pegará en Rayforge más abajo.

## Conectar Rayforge

El asistente de máquinas descubre los demonios en la red y lista sus dispositivos; elija uno para
añadirlo. Alternativamente, añada una máquina manualmente con el controlador **Remote machine
(raydriver)** e introduzca el nombre de host, el puerto y el token del demonio; luego elija el
dispositivo de la lista que se lee del demonio:

![Asistente — conexión remota](/screenshots/remote-wizard-connect.webp)

- **Nombre de host / puerto del demonio** — el ordenador que ejecuta el demonio (puerto por defecto
  `8765`).
- **Token** — el valor de `raydriver daemon status` en el host del dispositivo.
- **Dispositivo** — la máquina en el demonio, leída de su lista de dispositivos una vez configurados
  el nombre de host y el token.
- Los campos TLS solo aparecen para un demonio alcanzado por la red; un demonio en el mismo
  ordenador (`127.0.0.1`) no necesita certificado.

El sondeo funciona a través de la conexión remota, por lo que el asistente puede identificar
hardware que no está conectado a su ordenador.

## Ajustes de la máquina

Los ajustes de una máquina remota muestran el endpoint del demonio y el dispositivo en lugar de un
puerto serie local:

![Ajustes de la máquina — máquina remota](/screenshots/remote-settings.webp)

## Trabajos y reconexiones

- Un trabajo enviado se ejecuta hasta completarse en el host del dispositivo aunque Rayforge se
  cierre o la red se caiga.
- Reconecte para ver el progreso exacto. Un cliente a la vez tiene el control; los demás pueden
  observar.

## Cámaras en el host del dispositivo

El vídeo no viaja por la conexión de control. Use las [cámaras de red](camera.md) de Rayforge con un
stream accesible en el host del dispositivo:

- **Cámara IP**: añada su URL de stream directamente.
- **Cámara USB en el host del dispositivo**: ejecute un streamer estándar como `ustreamer` y añada
  su URL MJPEG como cámara de stream HTTP.

El demonio anuncia los endpoints de cámara configurados para que el asistente pueda rellenar la URL.

## Notas de seguridad

- Solo LAN (o túnel) — sin nube, sin cuentas.
- El token y TLS son obligatorios para cualquier demonio que no sea loopback.
- Mantenga los streams de cámara solo en LAN o tras un firewall; están separados del canal de
  control TLS.
