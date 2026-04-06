# Multi-Pasada

Multi-pasada repite la trayectoria de corte o grabado múltiples veces, opcionalmente bajando en Z entre pasadas. Esto es útil para materiales gruesos o crear grabados más profundos.

## Cómo Funciona

Cada pasada traza la misma trayectoria de nuevo. Con el descenso Z habilitado, el láser se mueve más cerca del material entre pasadas, cortando progresivamente más profundo.

## Ajustes

### Número de Pasadas

Cuántas veces repetir el paso completo (1-100). Cada pasada sigue la misma trayectoria.

- **1 pasada:** Corte único (por defecto)
- **2-3 pasadas:** Común para materiales medianamente gruesos
- **4+ pasadas:** Materiales muy gruesos o duros

### Descenso Z por Pasada

Distancia para bajar el eje Z entre pasadas (0-50 mm). Solo funciona si tu máquina tiene control de eje Z.

- **0 mm:** Todas las pasadas a la misma profundidad (por defecto)
- **Espesor del material ÷ pasadas:** Corte de profundidad progresiva
- **Incrementos pequeños (0.1-0.5mm):** Control fino para grabado profundo

:::warning Eje Z Requerido
El descenso Z solo funciona con máquinas que tienen control de eje Z motorizado. Para máquinas sin eje Z, todas las pasadas ocurren a la misma altura de enfoque.
:::

## Cuándo Usar Multi-Pasada

**Cortar materiales gruesos:**

Múltiples pasadas a la misma profundidad a menudo cortan más limpio que una sola pasada lenta. La primera pasada crea un kerf, y las pasadas subsiguientes siguen la misma trayectoria más eficientemente.

**Grabado profundo:**

Con descenso Z, puedes tallar patrones de relieve profundos o grabados que serían imposibles en una sola pasada.

**Mejor calidad de borde:**

Múltiples pasadas más rápidas a menudo producen bordes más limpios que una pasada lenta, especialmente en materiales que se queman fácilmente.

## Consejos

- Comienza con 2-3 pasadas a tu velocidad de corte normal
- Para materiales gruesos, aumenta las pasadas en lugar de reducir la velocidad
- Habilita el descenso Z solo si tu máquina lo soporta
- Prueba en material de desecho para encontrar el número óptimo de pasadas

---

## Temas Relacionados

- [Corte de Contorno](operations/contour) - Operación de corte principal
- [Grabado](operations/engrave) - Operaciones de grabado
