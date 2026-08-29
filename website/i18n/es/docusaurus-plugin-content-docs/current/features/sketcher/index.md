---
description: "Use el diseñador paramétrico 2D integrado de Rayforge para crear diseños láser listos con líneas, círculos, curvas Bézier y restricciones."
---

# Diseñador paramétrico 2D

El Diseñador paramétrico 2D es una función potente de Rayforge que le permite
crear y editar diseños 2D precisos basados en restricciones directamente dentro
de la aplicación. Esta función le permite diseñar piezas personalizadas desde
cero sin necesidad de software CAD externo.

## Descripción general

El diseñador proporciona un conjunto completo de herramientas para crear formas
geométricas y aplicar restricciones paramétricas para definir relaciones precisas
entre los elementos. Este enfoque garantiza que sus diseños mantengan la
geometría prevista incluso cuando se modifican las dimensiones.

## Creación y edición de bocetos

### Crear un nuevo boceto

1. Abra el panel inferior y haga clic en el botón **Nuevo boceto**, o haga clic
   derecho en el lienzo y seleccione **Nuevo boceto** en el menú contextual.
2. Se abrirá un nuevo espacio de trabajo vacío con la interfaz del editor de
   bocetos
3. Comience a crear geometría con las herramientas de dibujo del menú circular
   o los atajos de teclado
4. Aplique restricciones para definir las relaciones entre los elementos
5. Haga clic en "Finalizar boceto" para guardar su trabajo y volver al espacio
   de trabajo principal

### Editar bocetos existentes

1. Haga doble clic en una pieza de trabajo basada en boceto en el espacio de
   trabajo principal
2. Alternativamente, seleccione un boceto y elija "Editar boceto" en el menú
   contextual
3. Realice sus modificaciones con las mismas herramientas y restricciones
4. Haga clic en "Finalizar boceto" para guardar los cambios o en "Cancelar
   boceto" para descartarlos

## Consejos de flujo de trabajo

1. **Comience con geometría aproximada**: Cree primero formas básicas y luego
   refínelas con restricciones
2. **Use restricciones desde el principio**: Aplique restricciones mientras
   construye para mantener la intención del diseño
3. **Verifique el estado de las restricciones**: El sistema indica cuándo los
   bocetos están completamente restringidos
4. **Vigile los conflictos**: Las restricciones en conflicto se resaltan en rojo
   y se muestran en el panel de restricciones para facilitar su identificación
5. **Aproveche la simetría**: Las restricciones de simetría pueden acelerar
   significativamente los diseños complejos
6. **Use la cuadrícula**: Active la cuadrícula para una alineación precisa y use
   Ctrl para ajustar a la cuadrícula
7. **Itere y refínelo**: No dude en modificar las restricciones para obtener el
   resultado deseado

## Funciones de edición

- **Soporte completo de deshacer/rehacer**: El estado completo del boceto se
  guarda con cada operación
- **Cursor dinámico**: El cursor cambia para reflejar la herramienta de dibujo
  activa
- **Visualización de restricciones**: Las restricciones aplicadas se indican
  claramente en la interfaz
- **Actualizaciones en tiempo real**: Los cambios en las restricciones actualizan
  inmediatamente la geometría
- **Edición con doble clic**: Hacer doble clic en restricciones dimensionales
  (Distancia, Radio, Diámetro, Ángulo, Relación de aspecto) abre un diálogo
  para editar sus valores
- **Expresiones paramétricas**: Las restricciones dimensionales admiten
  expresiones, lo que permite calcular valores a partir de otros parámetros
  (p. ej., `width/2` para un radio que sea la mitad de la anchura)
