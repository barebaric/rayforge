# Configuração do Eixo Rotativo

O Rayforge suporta acessórios rotativos para gravar e cortar objetos cilíndricos
como canecas, copos, canetas e material redondo. Quando um módulo rotativo é
conectado, o Rayforge envolve o trabalho ao redor do cilindro e mostra uma pré-
visualização 3D do resultado.

![Configurações do Módulo Rotativo](/screenshots/machine-rotary-module.png)

## Quando Você Precisa do Modo Rotativo

Use o modo rotativo sempre que sua peça de trabalho for cilíndrica. Exemplos
comuns incluem:

- Gravar logotipos ou texto em artigos de bebida
- Cortar padrões em tubos ou canos
- Marcar objetos cilíndricos como canetas ou cabos de ferramentas

Sem o modo rotativo, o eixo Y move a cabeça do laser para frente e para trás em
uma cama plana. Com o modo rotativo ativado, o eixo Y controla a rotação do
cilindro, fazendo com que o desenho envolva a superfície.

## Configurando um Módulo Rotativo

Antes de começar, conecte fisicamente seu módulo rotativo à máquina de acordo com
as instruções do fabricante. Normalmente, isso significa conectá-lo à porta do
driver de passo do eixo Y no lugar do motor normal do eixo Y.

No Rayforge, abra **Configurações → Máquina** e navegue até a página **Rotativo**
para configurar seu módulo:

- **Circunferência**: Meça a distância ao redor do objeto que você deseja gravar.
  Você pode envolver um pedaço de papel ou barbante ao redor do cilindro e medir
  seu comprimento. Isso informa ao Rayforge o tamanho da superfície cilíndrica
  para que o desenho seja dimensionado corretamente.
- **Micropassos por rotação**: Este é o número de passos que o motor rotativo
  precisa para uma rotação completa. Consulte a documentação do seu módulo
  rotativo para encontrar este valor.

### Modos Rotativos

O Rayforge suporta dois modos rotativos:

- **4º eixo verdadeiro**: O rotativo opera como um quarto eixo independente junto com
  X, Y e Z. Este é o modo preferido quando seu controlador o suporta.
- **Substituição de eixo**: O rotativo assume o eixo Y ou Z. Este é o modo tradicional
  usado pela maioria dos controladores de hobby onde o rotativo é conectado a uma porta
  existente do driver de passo.

Você pode selecionar o modo na página de configurações do Rotativo.

### Acessórios Rotativos Tipo Rolamento

Acessórios rotativos tipo rolamento (onde o objeto repousa sobre rolos em vez de
preso em um mandril) têm sua própria página de configurações. Se seu rotativo usa
rolos em vez de um mandril, selecione o tipo de rolo na configuração rotativa e
insira os parâmetros do rolo.

## Modo Rotativo por Camada

Se o seu documento tiver várias camadas, você pode ativar ou desativar o modo
rotativo independentemente para cada camada. Isso é útil quando você deseja
combinar trabalho plano e cilíndrico em um único projeto, ou quando tem
configurações rotativas diferentes para partes diferentes do trabalho.

Quando o modo rotativo está ativo em uma camada, um pequeno ícone rotativo
aparece ao lado dessa camada na lista de camadas, para que você possa ver
rapidamente quais camadas serão executadas no modo rotativo.

## Pré-visualização 3D no Modo Rotativo

Quando o modo rotativo está ativo, a [visualização 3D](../ui/3d-preview) mostra
seu caminho de ferramenta envolvido ao redor de um cilindro em vez de uma
superfície plana. A tela 2D também se adapta automaticamente ao modo rotativo,
e a tela 3D renderiza os caminhos de ferramenta rotativos com precisão em todas
as configurações — incluindo durante a reprodução da simulação.

![Pré-visualização 3D no modo rotativo](/screenshots/main-3d-rotary.png)

Isso lhe dá uma pré-visualização realista de como o desenho ficará no objeto
real, facilitando a identificação de problemas de tamanho ou posicionamento
antes de começar a cortar.

### Modelo 3D do Módulo Rotativo

Cada módulo rotativo pode ter um modelo 3D atribuído. O modelo aparece na
[visualização 3D](../ui/3d-preview) junto com seu caminho de ferramenta, dando
a você uma melhor noção de como a configuração física fica. Você pode ajustar a
escala, posição e rotação do modelo para corresponder ao seu hardware real.

### Avanço em Z

O modo rotativo suporta avanço em Z, permitindo cortar progressivamente mais
fundo em peças cilíndricas ao longo de múltiplas passagens.

## Dicas para Bons Resultados

- **Meça a circunferência com cuidado** — mesmo um pequeno erro aqui vai
  esticar ou comprimir seu desenho ao redor do cilindro.
- **Fixe a peça de trabalho** — certifique-se de que o objeto esteja firmemente
  posicionado nos rolos e não balance ou escorregue durante o trabalho.
- **Teste primeiro com baixa potência** — faça uma passagem de gravação leve
  para verificar o alinhamento antes de se comprometer com um corte de potência
  total.
- **Mantenha a superfície limpa** — poeira ou resíduos no cilindro podem afetar
  a qualidade da gravação.

## Páginas Relacionadas

- [Fluxo de trabalho multicamadas](../features/multi-layer) - Configurações por
  camada incluindo rotativo
- [Visualização 3D](../ui/3d-preview) - Pré-visualizar caminhos de ferramenta em
  3D
- [Configurações da máquina](general) - Configuração geral da máquina
