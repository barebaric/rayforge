---
description: "Antes de ejecutar o exportar un trabajo, Rayforge verifica automáticamente problemas comunes como violaciones de límites, violaciones del área de trabajo y colisiones con zonas prohibidas."
---

# Comprobaciones de Sanity del Trabajo

Antes de ejecutar o exportar un trabajo, Rayforge realiza automáticamente un
conjunto de comprobaciones de sanity y presenta los resultados en un diálogo
estructurado. Esto te ayuda a detectar problemas temprano, antes de que se
conviertan en material arruinado.

![Diálogo de Comprobación de Sanity](/screenshots/sanity-check.png)

## Comprobaciones Realizadas

- **Violaciones de límites de la máquina**: Geometría que se extiende más allá
  de lo que tu máquina puede alcanzar físicamente, reportada por eje y dirección
- **Violaciones del área de trabajo**: Piezas de trabajo fuera de los límites
  del área de trabajo configurada
- **Colisiones con zonas prohibidas**: Trayectorias que pasan por zonas
  prohibidas habilitadas

Cada comprobación produce como máximo un problema por violación única, manteniendo
el diálogo legible incluso para proyectos complejos. El diálogo distingue entre
errores y advertencias, y puedes revisar todo antes de decidir si deseas proceder.

---

## Páginas Relacionadas

- [Zonas Prohibidas](../machine/nogo-zones) - Definir áreas restringidas en la
  superficie de trabajo
- [Vista 3D](../ui/3d-preview) - Visualización de trayectorias en 3D
