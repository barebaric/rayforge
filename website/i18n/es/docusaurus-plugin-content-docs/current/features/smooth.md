# Suavizar Trayectoria

El suavizado de trayectoria reduce los bordes irregulares y las transiciones bruscas en tus trayectorias de corte, resultando en curvas más limpias y movimiento de máquina más suave.

## Cómo Funciona

El suavizado aplica un filtro a la geometría de tu trayectoria que redondea las esquinas angulares y alisa los bordes ásperos. El láser sigue una trayectoria más suave en lugar de hacer cambios bruscos de dirección.

## Ajustes

### Habilitar Suavizado

Activa o desactiva el suavizado para esta operación. El suavizado está deshabilitado por defecto.

### Suavidad

Controla cuánto se suaviza la trayectoria (0-100). Valores más altos producen curvas más redondeadas pero pueden desviarse más de la trayectoria original.

- **Bajo (0-30):** Suavizado mínimo, preserva detalles finos
- **Medio (30-60):** Suavizado equilibrado para la mayoría de diseños
- **Alto (60-100):** Suavizado agresivo, mejor para formas orgánicas

### Umbral de Ángulo de Esquina

Ángulos más agudos que este valor se preservan como esquinas en lugar de suavizarse (0-179 grados). Esto evita que características afiladas importantes sean redondeadas.

- **Valores más bajos:** Más esquinas se suavizan, resultado más redondeado
- **Valores más altos:** Más esquinas se preservan, resultado más afilado

## Cuándo Usar Suavizado

**Bueno para:**

- Diseños importados de fuentes basadas en píxeles con efecto escalera
- Reducir el estrés mecánico en cambios rápidos de dirección
- Mejorar la calidad de corte en curvas
- Diseños con muchos segmentos de línea pequeños

**No necesario para:**

- Arte vectorial limpio con curvas bezier suaves
- Diseños donde las esquinas afiladas deben preservarse exactamente
- Dibujos técnicos que requieren geometría precisa

---

## Temas Relacionados

- [Corte de Contorno](operations/contour) - Operación de corte principal
- [Optimización de Trayectoria](path-optimization) - Reducir distancia de viaje
