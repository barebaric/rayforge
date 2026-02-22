# Soporte de Dialectos de Código G

Rayforge soporta múltiples dialectos de código G para trabajar con diferentes firmware de controlador.

## Dialectos Soportados

Rayforge actualmente soporta estos dialectos de código G:

| Dialecto                         | Firmware     | Uso Común                  | Estado                         |
| -------------------------------- | ------------ | -------------------------- | ------------------------------ |
| **GRBL (universal)**             | GRBL 1.1+    | Láseres de diodo, CNC de aficionado | ✓ Primario, completamente soportado |
| **GRBL (sin eje Z)**             | GRBL 1.1+    | Cortadores láser 2D sin Z  | ✓ Variante optimizada          |
| **GRBL Dinámico (Consciente de Profundidad)** | GRBL 1.1+    | Grabado láser consciente de profundidad | ✓ Recomendado para potencia dinámica |
| **GRBL Dinámico (sin eje Z)**    | GRBL 1.1+    | Grabado láser consciente de profundidad | ✓ Variante optimizada          |
| **Smoothieware**                 | Smoothieware | Cortadores láser, CNC      | ◐ Experimental                 |
| **Marlin**                       | Marlin 2.0+  | Impresoras 3D con láser    | ◐ Experimental                 |

:::note Dialectos Recomendados
:::

**GRBL (universal)** es el dialecto más probado y recomendado para aplicaciones láser estándar.

    **GRBL Dinámico (Consciente de Profundidad)** es recomendado para grabado láser consciente de profundidad donde la potencia varía durante los cortes (ej., grabado de profundidad variable).
---

## Creando un Dialecto Personalizado

Para crear un dialecto de código G personalizado basado en un dialecto integrado:

1. Abre **Ajustes de Máquina** → **Dialecto de Código G**
2. Haz clic en el ícono **Copiar** en un dialecto integrado para crear un nuevo dialecto personalizado
3. Edita los ajustes del dialecto según sea necesario
4. Guarda tu dialecto personalizado

Los dialectos personalizados se almacenan en tu directorio de configuración y pueden compartirse.

---

## Páginas Relacionadas

- [Exportando Código G](../files/exporting) - Ajustes de exportación
- [Compatibilidad de Firmware](firmware) - Versiones de firmware
- [Ajustes de Dispositivo](../machine/device) - Configuración de GRBL
- [Macros y Hooks](../machine/hooks-macros) - Inyección de código G personalizado
