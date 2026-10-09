# Limpeza de caminhos

Logotipos e desenhos vindos de outros programas nem sempre estão limpos. Um contorno pode terminar
uma fração de milímetro antes do ponto inicial, uma forma pode ser composta de várias partes
abertas, ou o mesmo caminho pode aparecer duas vezes no arquivo. O Rayforge corta esses caminhos
exatamente como estão: uma forma que só parece fechada é tratada como aberta, e um caminho duplicado
é cortado duas vezes.

As ferramentas de limpeza de caminhos corrigem isso nas peças de trabalho selecionadas. Você as
encontra no menu **Objeto** e em **Limpar caminhos** no menu de contexto da tela. Cada ferramenta é
um único passo de desfazer.

## Fechar caminhos

**Fechar caminhos…** fecha cada caminho aberto cujos pontos inicial e final estão mais próximos do
que a tolerância informada. A lacuna é preenchida com uma linha reta entre os dois pontos
existentes, então a forma não se move nem muda de tamanho.

## Unir caminhos abertos

**Unir caminhos abertos…** junta em caminhos mais longos os caminhos abertos cujas extremidades
estão mais próximas do que a tolerância. As partes são invertidas quando necessário, então a ordem e
o sentido em que foram desenhadas não importam. Quando as extremidades de um caminho unido passam a
se encontrar, o caminho também é fechado.

As duas ferramentas lembram a última tolerância usada durante a sessão. Uma tolerância entre 0,05 mm
e 0,2 mm funciona para a maioria dos arquivos importados.

## Excluir duplicados

**Excluir duplicados** remove caminhos que estão exatamente sobre outro caminho da mesma peça de
trabalho, independentemente do sentido ou do ponto inicial. Se você selecionar várias peças de
trabalho, uma peça que seja cópia exata de outra peça selecionada na mesma posição também é
removida.

Caminhos que se sobrepõem apenas em parte do comprimento são mantidos. O pós-processador
[Mesclar linhas](merge-lines) cuida deles quando o trabalho é gerado.

## Separar

**Separar** transforma cada caminho de uma peça de trabalho em uma peça própria, incluindo furos e
caminhos abertos. Isso é diferente de **Dividir**, que mantém uma ilha junto com seus furos.

A limpeza de caminhos só funciona em peças de trabalho importadas e vetorizadas. Peças que vêm de um
esboço são editadas no [Sketcher](sketcher/index).

## Páginas relacionadas

- [Mesclar linhas](merge-lines) - Cortar segmentos sobrepostos apenas uma vez
- [Ferramentas de tela](../ui/canvas-tools) - Selecionar e excluir segmentos individuais
- [Importando arquivos](../files/importing) - Trazer desenhos para o Rayforge
