# Zonas de Restrição

Zonas de restrição definem áreas restritas na superfície de trabalho que o laser
não deve entrar. Quando habilitadas, elas são verificadas como parte das
[verificações de sanidade do trabalho](../features/sanity-checks) antes de
executar ou exportar.

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

## Visibilidade

Zonas de restrição são exibidas tanto na tela 2D quanto na 3D como
sobreposições semi-transparentes. Use o botão de alternância de zonas de
restrição na sobreposição da tela para mostrá-las ou ocultá-las. A configuração
de visibilidade é lembrada entre sessões.

---

## Páginas Relacionadas

- [Configurações de Hardware](hardware) - Dimensões da máquina e configuração de eixos
- [Verificações de Sanidade do Trabalho](../features/sanity-checks) - Validação pré-trabalho
- [Visualização 3D](../ui/3d-preview) - Visualização de caminhos de ferramenta em 3D
