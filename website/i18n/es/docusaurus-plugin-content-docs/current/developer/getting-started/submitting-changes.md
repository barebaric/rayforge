# Enviando Cambios

Esta guía cubre el proceso para contribuir mejoras de código a Rayforge.

## Crear una Rama de Feature

Crea una rama descriptiva para tus cambios:

```bash
git checkout -b feature/nombre-de-tu-feature
# o
git checkout -b fix/numero-de-issue-descripcion
```

## Realiza Tus Cambios

- Sigue el estilo de código y convenciones existentes
- Escribe commits limpios y enfocados con mensajes claros
- Añade pruebas para nueva funcionalidad
- Actualiza documentación según sea necesario

## Prueba Tus Cambios

Ejecuta la suite de pruebas completa para asegurar que nada esté roto:

```bash
# Ejecutar todas las pruebas y linting
pixi run test
pixi run lint
```

## Sincroniza con Upstream

Antes de crear un pull request, sincroniza con el repositorio upstream:

```bash
# Obtener los últimos cambios
git fetch upstream

# Rebase tu rama en el último main
git rebase upstream/main
```

## Envía un Pull Request

1. Push tu rama a tu fork:

   ```bash
   git push origin feature/nombre-de-tu-feature
   ```

2. Crea un pull request en GitHub y rellena la plantilla de pull request:
   - Un título claro y una descripción del cambio y del problema que resuelve
   - Por qué elegiste tu enfoque arquitectónico concreto y qué alternativas descartaste
   - Los pasos exactos que seguiste para probar el cambio localmente
   - Referencia a cualquier issue relacionado
   - Capturas de pantalla si el cambio afecta la UI
   - Todas las casillas de la checklist de la plantilla marcadas

<!-- prettier-ignore-start -->
:::warning[La plantilla se aplica]
Los pull requests que no rellenan la plantilla se cierran automáticamente por un bot. Edita la
descripción del pull request para responder cada sección y marcar cada casilla; el pull request se
reabre automáticamente.
:::
<!-- prettier-ignore-end -->

## Proceso de Revisión de Código

- Todos los pull requests requieren revisión antes de merge
- Aborda feedback prontamente y realiza los cambios solicitados
- Mantén la discusión enfocada y constructiva

## Requisitos de Merge

Los pull requests se mergean cuando:

- [ ] Pasan todas las pruebas automatizadas
- [ ] Siguen el estilo de codificación del proyecto
- [ ] Incluyen pruebas apropiadas para nueva funcionalidad
- [ ] Tienen actualizaciones de documentación si es necesario
- [ ] Son aprobados por al menos un maintainer

## Código Asistido por IA

Agradecemos el uso de herramientas de IA para ayudar a escribir código. Sin embargo, eres
estrictamente responsable del código que envías. DEBES entenderlo completamente y debes poder
explicar las decisiones arquitectónicas.

El código generado por IA a menudo introduce errores sutiles, desajustes arquitectónicos y fallos de
seguridad. Los pull requests que parezcan salida de IA sin revisar se cerrarán para respetar el
tiempo de los maintainers.

## Directrices Adicionales

### Mensajes de Commit

Usa mensajes de commit claros y descriptivos:

- Comienza con una letra mayúscula
- Mantén la primera línea bajo 50 caracteres
- Usa el modo imperativo ("Añadir feature" no "Añadida feature")
- Incluye más detalle en el cuerpo si es necesario

### Cambios Pequeños y Enfocados

Mantén los pull requests enfocados en una sola feature o fix. Los cambios grandes deberían dividirse
en piezas más pequeñas y lógicas.

<!-- prettier-ignore-start -->
:::tip[Discute Primero]
Para cambios importantes, abre un
[issue](https://github.com/barebaric/rayforge/issues) primero para discutir tu enfoque antes de
invertir tiempo significativo.
:::
<!-- prettier-ignore-end -->

<!-- prettier-ignore-start -->
:::note[¿Necesitas Ayuda?]
Si no estás seguro sobre alguna parte del proceso de contribución, no
dudes en pedir ayuda en un issue o discusión.
:::
<!-- prettier-ignore-end -->
