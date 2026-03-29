# Fusionar líneas

Al importar un diseño que contiene trazados superpuestos, el láser puede
terminar cortando la misma línea más de una vez. Esto desperdicia tiempo,
puede causar un chamuscado excesivo y ensanchar el kerf más de lo deseado.

El postprocesador **Fusionar líneas** detecta los segmentos de trazado
superpuestos y coincidentes, y los fusiona en un solo paso. El láser recorre
cada línea única solamente una vez.

## Cuándo usarlo

Esto ocurre con mayor frecuencia cuando:

- Importa un SVG o DXF donde las figuras comparten bordes (por ejemplo, un
  patrón de cuadrícula o teselación)
- Combina múltiples piezas de trabajo cuyos contornos se superponen
- Su software de diseño exporta trazados duplicados

## Cuándo no usarlo

Si los cortes superpuestos son intencionales — por ejemplo, hacer múltiples
pasadas sobre la misma línea para cortar material más grueso — deje la opción
Fusionar líneas desactivada. En ese caso, es posible que desee usar la función
[Multipasada](multi-pass), que le da un control explícito sobre la cantidad
de pasadas.

## Páginas relacionadas

- [Optimización de trazados](path-optimization) - Reducción de movimientos de
  desplazamiento innecesarios
- [Multipasada](multi-pass) - Múltiples pasadas intencionales sobre el mismo
  trazado
- [Corte de contorno](operations/contour) - La operación principal de corte
