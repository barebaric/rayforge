# Compensación de Kerf

Kerf es el material removido por el haz láser durante el corte. La compensación de kerf ajusta las trayectorias de herramienta para considerar esto, asegurando que las piezas cortadas coincidan con sus dimensiones diseñadas.

## ¿Qué es Kerf?

**Kerf** = el ancho de material removido por el proceso de corte.

**Ejemplo:**
- Tamaño de punto láser: 0.2mm
- Interacción con material: añade ~0.1mm en cada lado
- **Kerf total:** ~0.4mm

---

## Cómo Funciona la Compensación de Kerf

La compensación de kerf **desplaza la trayectoria de herramienta** hacia adentro o hacia afuera para considerar la remoción de material:

**Para cortes exteriores (cortar una pieza):**
- Desplazar trayectoria **hacia afuera** por la mitad del ancho de kerf
- Resultado: La pieza final tiene el tamaño correcto

**Para cortes interiores (cortar un agujero):**
- Desplazar trayectoria **hacia adentro** por la mitad del ancho de kerf
- Resultado: El agujero final tiene el tamaño correcto

**Ejemplo con kerf de 0.4mm:**

```
Trayectoria original:  cuadrado de 50mm
Compensación:          Desplazar hacia afuera por 0.2mm (medio kerf)
El láser sigue:        cuadrado de 50.4mm
Después de cortar:     La pieza mide 50.0mm (¡perfecto!)
```

---

## Midiendo Kerf

**Procedimiento de medición precisa de kerf:**

1. **Crea un archivo de prueba:**
   - Dibuja un cuadrado de 50mm x 50mm
   - Dibuja un círculo (cualquier tamaño, para prueba de corte interior)

2. **Corta la prueba:**
   - Usa tus ajustes de corte normales
   - Corta completamente a través
   - Deja que el material se enfríe

3. **Mide:**
   - **Cuadrado exterior (pieza):** Mide con calibrador
     - Si < 50mm, el kerf fue removido hacia afuera
     - Kerf = (50 - medido) x 2
   - **Círculo interior (agujero):** Mide el diámetro
     - Si > diámetro diseñado, el kerf fue removido hacia adentro
     - Kerf = (medido - diseñado) / 2

4. **Promedia:** Usa el promedio de múltiples mediciones

**Variables que afectan el kerf:**
- Potencia del láser (mayor = más ancho)
- Velocidad de corte (más lento = más ancho)
- Tipo y densidad del material
- Distancia de enfoque
- Presión de asistencia de aire

---

## Compensación de Kerf Manual

Si la compensación de kerf automatizada no está disponible, compensa en tu software de diseño:

**Inkscape:**

1. **Selecciona la trayectoria**
2. **Trayectoria → Desplazamiento Dinámico** (Ctrl+J)
3. **Arrastra para desplazar** por la mitad de tu medición de kerf
   - Hacia afuera para piezas (para hacer la trayectoria más grande)
   - Hacia adentro para agujeros (para hacer la trayectoria más pequeña)
4. **Trayectoria → Objeto a Trayectoria** para finalizar

**Illustrator:**

1. **Selecciona la trayectoria**
2. **Objeto → Trayectoria → Desplazar Trayectoria**
3. **Ingresa valor de desplazamiento:** (kerf / 2)
   - Positivo para hacia afuera, negativo para hacia adentro
4. **OK** para aplicar

**Fusion 360 / CAD:**

- Desplaza entidades de boceto antes de exportar
- Usa la dimensión de kerf/desplazamiento

---

## Páginas Relacionadas

- [Operación de Contorno](operations/contour) - Operaciones de corte
- [Cuadrícula de Prueba de Material](operations/material-test-grid) - Encontrar ajustes óptimos
