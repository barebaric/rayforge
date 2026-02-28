# Permisos de Snap (Linux)

Esta página explica cómo configurar permisos para Rayforge cuando se instala como paquete Snap en Linux.

## ¿Qué son los Permisos de Snap?

Los Snaps son aplicaciones en contenedores que se ejecutan en un sandbox por seguridad. Por defecto, tienen acceso limitado a los recursos del sistema. Para usar ciertas funciones (como puertos serie para controladores láser), debes otorgar permisos explícitamente.

## Permisos Requeridos

Rayforge necesita estas interfaces de Snap conectadas para funcionalidad completa:

| Interfaz     | Propósito                                         | ¿Requerido? |
| ------------ | ------------------------------------------------- | ------------ |
| `serial-port`| Acceso a dispositivos serie USB (controladores láser) | **Sí** (para control de máquina) |
| `home`       | Leer/escribir archivos en tu directorio personal  | Auto-conectado |
| `removable-media` | Acceso a unidades externas y almacenamiento USB | Opcional |
| `network`    | Conectividad de red (para actualizaciones, etc.)  | Auto-conectado |

---

## Otorgando Acceso al Puerto Serie

**Este es el permiso más importante para Rayforge.**

### Prerrequisito: membresía al grupo dialout

En distribuciones basadas en Debian, tu usuario debe ser miembro del grupo
`dialout`, incluso cuando uses el paquete Snap. Sin esta membresía al grupo,
puedes recibir mensajes AppArmor DENIED al intentar acceder a los puertos
serie.

```bash
# Añade tu usuario al grupo dialout
sudo usermod -a -G dialout $USER
```

**Importante:** Debes cerrar sesión y volver a entrar (o reiniciar) para que
los cambios de grupo surtan efecto.

### Verificar Permisos Actuales

```bash
# Ver todas las conexiones para Rayforge
snap connections rayforge
```

Busca la interfaz `serial-port`. Si muestra "desconectado" o "-", necesitas conectarla.

### Conectar Interfaz de Puerto Serie

```bash
# Otorgar acceso al puerto serie
sudo snap connect rayforge:serial-port
```

**Solo necesitas hacer esto una vez.** El permiso persiste a través de actualizaciones de la app y reinicios.

### Verificar Conexión

```bash
# Verificar si serial-port ahora está conectado
snap connections rayforge | grep serial-port
```

Salida esperada:
```
serial-port     rayforge:serial-port     :serial-port     -
```

Si ves un indicador de plug/slot, la conexión está activa.

---

## Otorgando Acceso a Medios Removibles

Si quieres importar/exportar archivos desde unidades USB o almacenamiento externo:

```bash
# Otorgar acceso a medios removibles
sudo snap connect rayforge:removable-media
```

Ahora puedes acceder a archivos en `/media` y `/mnt`.

---

## Solución de Problemas de Permisos de Snap

### El Puerto Serie Todavía No Funciona

**Después de conectar la interfaz:**

1. **Vuelve a conectar el dispositivo USB:**
   - Desconecta tu controlador láser
   - Espera 5 segundos
   - Vuelve a conectarlo

2. **Reinicia Rayforge:**
   - Cierra Rayforge completamente
   - Relanza desde el menú de aplicaciones o:
     ```bash
     snap run rayforge
     ```

3. **Verifica que el puerto aparezca:**
   - Abre Rayforge → Configuración → Máquina
   - Busca puertos serie en el menú desplegable
   - Deberías ver `/dev/ttyUSB0`, `/dev/ttyACM0`, o similar

4. **Verifica que el dispositivo existe:**
   ```bash
   # Listar dispositivos serie USB
   ls -l /dev/ttyUSB* /dev/ttyACM*
   ```

### "Permiso Denegado" A Pesar de la Interfaz Conectada

Esto es raro pero puede suceder si:

1. **La instalación de Snap está rota:**
   ```bash
   # Reinstalar el snap
   sudo snap refresh rayforge --devmode
   # O si eso falla:
   sudo snap remove rayforge
   sudo snap install rayforge
   # Volver a conectar interfaces
   sudo snap connect rayforge:serial-port
   ```

2. **Reglas udev en conflicto:**
   - Revisa `/etc/udev/rules.d/` para reglas personalizadas de puerto serie
   - Podrían estar en conflicto con el acceso a dispositivos de Snap

3. **Denegaciones de AppArmor:**
   ```bash
   # Verificar denegaciones de AppArmor
   sudo journalctl -xe | grep DENIED | grep rayforge
   ```

   Si ves denegaciones para puertos serie, puede haber un conflicto de perfil de AppArmor.

### No Puedo Acceder a Archivos Fuera del Directorio Personal

**Por diseño**, los Snaps no pueden acceder a archivos fuera de tu directorio personal a menos que otorgues `removable-media`.

**Opciones alternativas:**

1. **Mueve archivos a tu directorio personal:**
   ```bash
   # Copia archivos SVG a ~/Documentos
   cp /alguna/otra/ubicacion/*.svg ~/Documentos/
   ```

2. **Otorga acceso a removable-media:**
   ```bash
   sudo snap connect rayforge:removable-media
   ```

3. **Usa el selector de archivos de Snap:**
   - El selector de archivos integrado tiene acceso más amplio
   - Abre archivos a través de Archivo → Abrir en lugar de argumentos de línea de comandos

---

## Gestión Manual de Interfaces

### Listar Todas las Interfaces Disponibles

```bash
# Ver todas las interfaces de Snap en tu sistema
snap interface
```

### Desconectar una Interfaz

```bash
# Desconectar serial-port (si es necesario)
sudo snap disconnect rayforge:serial-port
```

### Reconectar Después de Desconectar

```bash
sudo snap connect rayforge:serial-port
```

---

## Alternativa: Instalar desde Código Fuente

Si los permisos de Snap son demasiado restrictivos para tu flujo de trabajo:

**Opción 1: Compilar desde código fuente**

```bash
# Clonar el repositorio
git clone https://github.com/kylemartin57/rayforge.git
cd rayforge

# Instalar dependencias usando pixi
pixi install

# Ejecutar Rayforge
pixi run rayforge
```

**Beneficios:**
- Sin restricciones de permisos
- Acceso completo al sistema
- Depuración más fácil
- Última versión de desarrollo

**Desventajas:**
- Actualizaciones manuales (git pull)
- Más dependencias para gestionar
- Sin actualizaciones automáticas

**Opción 2: Usar Flatpak (si está disponible)**

Flatpak tiene sandboxing similar pero a veces con diferentes modelos de permisos. Verifica si Rayforge ofrece un paquete Flatpak.

---

## Mejores Prácticas de Permisos de Snap

### Solo Conecta Lo Que Necesitas

No conectes interfaces que no usas:

- ✓ Conecta `serial-port` si usas un controlador láser
- ✓ Conecta `removable-media` si importas desde unidades USB
- ✗ No conectes todo "por si acaso" - derrota el propósito de seguridad

### Verifica la Fuente del Snap

Siempre instala desde la Snap Store oficial:

```bash
# Verificar editor
snap info rayforge
```

Busca:
- Editor verificado
- Fuente de repositorio oficial
- Actualizaciones regulares

---

## Entendiendo el Sandbox de Snap

### ¿Qué Pueden Acceder los Snaps por Defecto?

**Permitido:**
- Archivos en tu directorio personal
- Conexiones de red
- Pantalla/audio

**No permitido sin permiso explícito:**
- Puertos serie (dispositivos USB)
- Medios removibles
- Archivos del sistema
- Directorios personales de otros usuarios

### Por Qué Esto Importa para Rayforge

Rayforge necesita:

1. **Acceso al directorio personal** (otorgado automáticamente)
   - Para guardar archivos de proyecto
   - Para leer archivos SVG/DXF importados
   - Para almacenar preferencias

2. **Acceso al puerto serie** (debe otorgarse)
   - Para comunicarse con controladores láser
   - **Este es el permiso crítico**

3. **Medios removibles** (opcional)
   - Para importar archivos desde unidades USB
   - Para exportar código G a almacenamiento externo

---

## Depurando Problemas de Snap

### Habilitar Registro Verbose de Snap

```bash
# Ejecutar Snap con salida de depuración
snap run --shell rayforge
# Dentro del shell de snap:
export RAYFORGE_LOG_LEVEL=DEBUG
exec rayforge
```

### Verificar Registros de Snap

```bash
# Ver registros de Rayforge
snap logs rayforge

# Seguir registros en tiempo real
snap logs -f rayforge
```

### Verificar Diario del Sistema para Denegaciones

```bash
# Buscar denegaciones de AppArmor
sudo journalctl -xe | grep DENIED | grep rayforge

# Buscar eventos de dispositivo USB
sudo journalctl -f -u snapd
# Luego conecta tu controlador láser
```

---

## Obteniendo Ayuda

Si todavía tienes problemas relacionados con Snap:

1. **Verifica permisos primero:**
   ```bash
   snap connections rayforge
   ```

2. **Prueba el puerto serie:**
   ```bash
   # Si tienes screen o minicom instalado
   sudo snap connect rayforge:serial-port
   # Luego prueba en Rayforge
   ```

3. **Reporta el problema con:**
   - Salida de `snap connections rayforge`
   - Salida de `snap version`
   - Salida de `snap info rayforge`
   - Tu versión de distribución Ubuntu/Linux
   - Mensajes de error exactos

4. **Considera alternativas:**
   - Instalar desde código fuente (ver arriba)
   - Usar un formato de paquete diferente (AppImage, Flatpak)

---

## Comandos de Referencia Rápida

```bash
# Otorgar acceso al puerto serie (más importante)
sudo snap connect rayforge:serial-port

# Otorgar acceso a medios removibles
sudo snap connect rayforge:removable-media

# Verificar conexiones actuales
snap connections rayforge

# Ver registros de Rayforge
snap logs rayforge

# Actualizar/refrescar Rayforge
sudo snap refresh rayforge

# Eliminar y reinstalar (último recurso)
sudo snap remove rayforge
sudo snap install rayforge
sudo snap connect rayforge:serial-port
```

---

## Páginas Relacionadas

- [Problemas de Conexión](connection) - Solución de problemas de conexión serie
- [Modo Depuración](debug) - Habilitar registro de diagnóstico
- [Instalación](../getting-started/installation) - Guía de instalación
- [Ajustes Generales](../machine/general) - Configuración de máquina
