# Mesclar linhas

Ao importar um projeto que contém caminhos sobrepostos, o laser pode acabar
cortando a mesma linha mais de uma vez. Isso desperdiça tempo, pode causar
carbonização excessiva e alargar o kerf além do pretendido.

O pós-processador **Mesclar linhas** detecta segmentos de caminho sobrepostos
e coincidentes, mesclando-os em um único passe. O laser segue cada linha única
apenas uma vez.

## Quando usar

Isso ocorre com mais frequência quando:

- Você importa um SVG ou DXF onde as formas compartilham bordas (por exemplo,
  um padrão de grade ou mosaico)
- Você combina múltiplas peças de trabalho cujos contornos se sobrepõem
- Seu software de design exporta caminhos duplicados

## Quando não usar

Se os cortes sobrepostos forem intencionais — por exemplo, fazer múltiplos
passes sobre a mesma linha para cortar material mais espesso — deixe a opção
Mesclar linhas desativada. Nesse caso, você pode usar o recurso
[Passagem múltipla](multi-pass), que oferece controle explícito sobre o número
de passes.

## Páginas relacionadas

- [Otimização de caminho](path-optimization) - Redução de movimentos de
  deslocamento desnecessários
- [Passagem múltipla](multi-pass) - Passagens intencionais múltiplas sobre o
  mesmo caminho
- [Corte de contorno](operations/contour) - A operação de corte principal
