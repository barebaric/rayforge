# Zonas Prohibidas

Las zonas prohibidas definen áreas restringidas en la superficie de trabajo que el láser
no debería ingresar. Antes de ejecutar o exportar un trabajo, Rayforge verifica si alguna
trayectoria ingresa a una zona prohibida habilitada y te advierte si se detecta una colisión.

![Zonas Prohibidas](/screenshots/machine-nogo-zones.png)

## Añadir una Zona Prohibida

Abre **Ajustes → Máquina** y navega a la página **Zonas Prohibidas**. Haz clic
en el botón añadir para crear una nueva zona, luego elige su forma y posición.

Cada zona tiene los siguientes ajustes:

- **Nombre**: Una etiqueta descriptiva para la zona
- **Habilitado**: Activa o desactiva la zona sin eliminarla
- **Forma**: Rectángulo, Caja o Cilindro
- **Posición (X, Y, Z)**: Dónde se coloca la zona en la superficie de trabajo
- **Dimensiones**: Ancho, alto y profundidad (o radio para cilindros)

## Advertencias de Colisión

Cuando ejecutas o exportas un trabajo, Rayforge verifica todas las trayectorias contra
las zonas prohibidas habilitadas. Si una trayectoria pasa por una zona, aparece un
diálogo de advertencia con la opción de cancelar o proceder bajo tu propio riesgo.

## Visibilidad

Las zonas prohibidas se muestran tanto en el lienzo 2D como en el 3D como
superposiciones semi-transparentes. Usa el botón de alternar zonas prohibidas en la
superposición del lienzo para mostrarlas u ocultarlas. El ajuste de visibilidad se
recuerda entre sesiones.

---

## Páginas Relacionadas

- [Ajustes de Hardware](hardware) - Dimensiones de máquina y configuración de ejes
- [Vista 3D](../ui/3d-preview) - Visualización de trayectorias en 3D
