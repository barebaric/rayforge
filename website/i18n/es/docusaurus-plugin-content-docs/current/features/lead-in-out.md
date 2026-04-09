# Entrada / Salida

Los movimientos de entrada y salida extienden cada trayectoria de contorno con segmentos cortos sin potencia antes de que comience el corte y después de que termine. Esto le da tiempo a la cabeza láser para alcanzar una velocidad constante antes de que comience el corte real y para desacelerar gradualmente después de que el corte termine, lo que produce resultados más limpios en los puntos de inicio y fin de cada corte.

## Cómo Funciona

Cuando la entrada/salida está habilitada, Rayforge observa la dirección tangente de cada trayectoria de contorno en sus puntos de inicio y fin. Luego inserta un movimiento recto corto sin potencia del láser a lo largo de esa tangente antes del primer punto de corte y otro después del último punto de corte. El láser está apagado durante estos segmentos adicionales, por lo que no se elimina material fuera de la trayectoria prevista.

## Ajustes

### Habilitar Entrada/Salida

Activa o desactiva la función para la operación. Cuando está deshabilitado, el corte comienza y termina exactamente en los puntos finales de la trayectoria sin movimientos adicionales de aproximación o salida.

### Distancia Automática

Cuando esta opción está habilitada, Rayforge calcula automáticamente la distancia de entrada y salida basándose en la velocidad de corte y el ajuste de aceleración de la máquina. La fórmula utiliza un factor de seguridad de dos para asegurar que la cabeza láser tenga suficiente espacio para alcanzar la velocidad completa. Cada vez que cambias la velocidad de corte o se actualiza la aceleración de la máquina, la distancia se recalcula.

### Distancia de Entrada

La longitud del movimiento de aproximación sin potencia antes de que comience el corte, en milímetros. El valor predeterminado es 2 mm. Este campo solo es editable cuando la distancia automática está desactivada.

### Distancia de Salida

La longitud del movimiento de salida sin potencia después de que termina el corte, en milímetros. El valor predeterminado es 2 mm. Este campo solo es editable cuando la distancia automática está desactivada.

## Cuándo Usar Entrada/Salida

La entrada/salida es más útil cuando notas marcas de quemadura, sobre-quemado o calidad de corte inconsistente en los puntos de inicio y fin de tus contornos. La aproximación sin potencia le da tiempo a la máquina para acelerar a la velocidad de corte, de modo que el láser alcance el material a velocidad completa, y la salida sin potencia permite una desaceleración suave en lugar de permanecer a potencia completa en el último punto.

Está disponible como opción de postprocesamiento en operaciones de contorno, contorno de marco y shrink wrap.

---

## Páginas Relacionadas

- [Corte de Contorno](operations/contour) - Operación de corte principal
- [Contorno de Marco](operations/frame-outline) - Corte de límite rectangular
- [Shrink Wrap](operations/shrink-wrap) - Corte de límite eficiente
- [Pestañas de Sujeción](holding-tabs) - Mantener las piezas seguras durante el corte
