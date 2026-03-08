# Recortar al Material

Recortar al Material limita las trayectorias de corte al límite de tu material. Cualquier corte que se extienda más allá del área del material se recorta, evitando que el láser corte fuera de tu material.

## Cómo Funciona

El transformador compara tus trayectorias de corte contra el límite definido del material. Los segmentos de trayectoria fuera de este límite se eliminan o recortan al borde del material.

## Ajustes

### Habilitar Recortar al Material

Activa o desactiva el recorte. Deshabilitado por defecto.

### Desplazamiento

Ajusta el límite efectivo del material antes de recortar (-100 a +100 mm).

- **Valores positivos:** Reduce el límite (corta más conservadoramente)
- **Valores negativos:** Expande el límite (permite cortes más cerca del borde)
- **0 mm:** Usa el límite exacto del material

Usa el desplazamiento cuando quieres un margen de seguridad desde el borde del material, o cuando la colocación de tu material no está perfectamente alineada.

## Cuándo Usar Recortar al Material

**Diseños parciales:** Tu diseño es más grande que tu material, pero quieres cortar solo la porción que cabe.

**Margen de seguridad:** Previene cortes accidentales más allá de los bordes del material.

**Láminas anidadas:** Corta solo las partes que caben en tu pieza actual de material.

**Cortes de prueba:** Limita una prueba a un área específica de tu material.

## Ejemplo

Tienes un diseño grande pero solo una pieza pequeña de material:

1. Define el tamaño de tu material para que coincida con tu material
2. Habilita Recortar al Material
3. Configura el desplazamiento a 2mm para margen de seguridad
4. Solo las porciones dentro del límite de tu material serán cortadas

---

## Temas Relacionados

- [Manejo de Material](stock-handling) - Configurar límites de material
- [Corte de Contorno](operations/contour) - Operación de corte principal
