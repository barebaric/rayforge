# Exportando desde Rayforge

Rayforge soporta varias opciones de exportación para diferentes propósitos:

- **Código G** - Salida de control de máquina para ejecutar trabajos
- **Exportar Objeto** - Exportar piezas individuales a formatos vectoriales
- **Exportar Documento** - Exportar todas las piezas como un solo archivo

---

## Exportando Objetos

Puedes exportar cualquier pieza a formatos vectoriales para usar en software de diseño, aplicaciones
CAD, o para archivar.

### Cómo Exportar

1. **Selecciona una pieza** en el lienzo
2. **Elige Objeto → Exportar Objeto...** (o clic derecho → Exportar Objeto...)
3. **Selecciona formato** y ubicación de guardado

### Formatos Disponibles

| Formato  | Extensión | Descripción                                                                                                    |
| ------- | --------- | -------------------------------------------------------------------------------------------------------------- |
| **RFS** | `.rfs`    | Formato de bosquejo paramétrico nativo de Rayforge. Preserva todas las restricciones y puede reimportarse para edición. |
| **SVG** | `.svg`    | Gráficos Vectoriales Escalables. Ampliamente compatible con software de diseño como Inkscape, Illustrator y navegadores web. |
| **DXF** | `.dxf`    | Formato de Intercambio de Dibujos. Compatible con la mayoría de aplicaciones CAD como AutoCAD, FreeCAD y LibreCAD. |

### Notas de Exportación

- **SVG y DXF** exportan la geometría resuelta (no restricciones paramétricas)
- Las exportaciones usan **unidades de milímetros**
- La geometría se escala a dimensiones reales (espacio del mundo)
- Múltiples subtrayectorias (formas desconectadas) se preservan como elementos separados

### Casos de Uso

**Compartiendo diseños:**

- Exporta a SVG para compartir con usuarios de Inkscape
- Exporta a DXF para usuarios de software CAD

**Edición posterior:**

- Exporta a SVG/DXF, edita en software externo, reimporta

**Archivando:**

- Usa RFS para diseños basados en bosquejo para preservar editabilidad
- Usa SVG/DXF para almacenamiento a largo plazo o usuarios sin Rayforge

---

## Exportando Documentos

Puedes exportar todas las piezas en un documento a un solo archivo vectorial. Esto es
útil para compartir proyectos completos o crear respaldos en formatos estándar.

### Cómo Exportar

1. **Elige Archivo → Exportar Documento...**
2. **Selecciona formato** (SVG o DXF)
3. **Elige ubicación de guardado**

### Formatos Disponibles

| Formato  | Extensión | Descripción                                                                                                    |
| ------- | --------- | -------------------------------------------------------------------------------------------------------------- |
| **SVG** | `.svg`    | Gráficos Vectoriales Escalables. Ampliamente compatible con software de diseño como Inkscape, Illustrator y navegadores web. |
| **DXF** | `.dxf`    | Formato de Intercambio de Dibujos. Compatible con la mayoría de aplicaciones CAD como AutoCAD, FreeCAD y LibreCAD. |

### Notas de Exportación

- Todas las piezas de todas las capas se combinan en un solo archivo
- Las posiciones de las piezas se preservan
- Las piezas vacías se omiten
- El cuadro delimitador abarca toda la geometría

### Casos de Uso

- **Compartir proyecto**: Exporta proyecto completo para colaboración
- **Respaldo**: Crea un archivo visual de tu trabajo
- **Edición posterior**: Lleva todo el diseño a Inkscape o software CAD

---

## Exportando Código G

El código G generado contiene todo exactamente como se enviaría a la máquina.
El formato exacto, comandos, precisión numérica, etc. dependen de los ajustes de
la máquina actualmente seleccionada y su dialecto de código G.

---

### Métodos de Exportación

### Método 1: Menú Archivo

**Archivo Exportar Código G** (Ctrl+E)

- Abre diálogo de guardar archivo
- Elige ubicación y nombre de archivo
- Código G generado y guardado

### Método 2: Línea de Comandos

```bash
# Exportar desde línea de comandos (si es compatible)
rayforge --export output.gcode input.svg
```

---

### Salida de Código G

El código G generado contiene todo exactamente como se enviaría a la máquina.
El formato exacto, comandos, precisión numérica, etc. dependen de los ajustes de
la máquina actualmente seleccionada y su dialecto de código G.

---

## Páginas Relacionadas

- [Importando Archivos](importing) - Obteniendo diseños en Rayforge
- [Formatos Soportados](formats) - Detalles de formatos de archivo
- [Dialectos de Código G](../reference/gcode-dialects) - Diferencias de dialectos
- [Hooks y Macros](../machine/hooks-macros) - Personalizando la salida
- [Modo Simulación](../features/simulation-mode) - Previsualizar antes de exportar
