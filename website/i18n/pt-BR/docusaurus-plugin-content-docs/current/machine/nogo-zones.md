# Zonas de Restrição

Zonas de restrição definem áreas restritas na superfície de trabalho que o laser
não deve entrar. Antes de executar ou exportar um trabalho, o Rayforge verifica
se algum caminho de ferramenta entra em uma zona de restrição habilitada e avisa
você se uma colisão é detectada.

![Zonas de Restrição](/screenshots/machine-nogo-zones.png)

## Adicionando uma Zona de Restrição

Abra **Configurações → Máquina** e navegue até a página **Zonas de Restrição**.
Clique no botão adicionar para criar uma nova zona, depois escolha sua forma e
posição.

Cada zona tem as seguintes configurações:

- **Nome**: Um rótulo descritivo para a zona
- **Habilitado**: Ativar ou desativar a zona sem excluí-la
- **Forma**: Retângulo, Caixa ou Cilindro
- **Posição (X, Y, Z)**: Onde a zona está colocada na superfície de trabalho
- **Dimensões**: Largura, altura e profundidade (ou raio para cilindros)

## Avisos de Colisão

Quando você executa ou exporta um trabalho, o Rayforge verifica todos os
caminhos de ferramenta contra as zonas de restrição habilitadas. Se um caminho
de ferramenta passa por uma zona, um diálogo de aviso aparece com a opção de
cancelar ou prosseguir por sua conta e risco.

## Visibilidade

Zonas de restrição são exibidas tanto na tela 2D quanto na 3D como
sobreposições semi-transparentes. Use o botão de alternância de zonas de
restrição na sobreposição da tela para mostrá-las ou ocultá-las. A configuração
de visibilidade é lembrada entre sessões.

---

## Páginas Relacionadas

- [Configurações de Hardware](hardware) - Dimensões da máquina e configuração de eixos
- [Visualização 3D](../ui/3d-preview) - Visualização de caminhos de ferramenta em 3D
