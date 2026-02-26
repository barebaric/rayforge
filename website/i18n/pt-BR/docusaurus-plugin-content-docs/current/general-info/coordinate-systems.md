# Sistemas de Coordenadas de Trabalho (WCS)

Os Sistemas de Coordenadas de Trabalho (WCS) permitem que você defina múltiplos pontos de referência na área de trabalho da sua máquina. Isso facilita executar o mesmo trabalho em diferentes posições sem redesenhar ou reposicionar suas peças.

## Espaços de Coordenadas

O Rayforge usa três espaços de coordenadas que trabalham juntos:

| Espaço       | Descrição                                                                                                             |
| ------------ | --------------------------------------------------------------------------------------------------------------------- |
| **MACHINE**  | Coordenadas absolutas relativas à posição de origem da máquina. A origem é fixada pelo hardware.                      |
| **WORKAREA** | A área utilizável dentro da sua máquina, considerando as margens ao redor da mesa.                                    |
| **WCS**      | O sistema de coordenadas do seu trabalho. Origem configurável pelo usuário para design e posicionamento de trabalhos. |

:::note Nota para Desenvolvedores
Internamente, o Rayforge usa um sistema de coordenadas normalizado chamado espaço WORLD.
O espaço WORLD descreve o mesmo espaço físico que o espaço MACHINE, mas com uma
convenção fixa: Y para cima com origem no canto inferior esquerdo. Isso simplifica
cálculos internos e renderização. Os usuários não precisam interagir diretamente
com o espaço WORLD.
:::

### Espaço MACHINE

O espaço MACHINE é o sistema de coordenadas absoluto relativo à posição de origem
da sua máquina. A origem (0,0) é determinada pela configuração de homing da sua máquina.

- **Origem**: Posição de origem da máquina (0,0,0) - fixada pelo hardware
- **Propósito**: Referência para todos os outros sistemas de coordenadas
- **Fixo**: Não pode ser alterado por software

A direção das coordenadas depende da configuração da sua máquina:

- **Canto de origem**: Pode ser superior-esquerdo, inferior-esquerdo, superior-direito ou inferior-direito
- **Direção dos eixos**: Os eixos X e Y podem ser invertidos com base na configuração do hardware

### Espaço WORKAREA

O espaço WORKAREA define a área utilizável dentro da sua máquina, considerando
quaisquer margens ao redor das bordas da sua mesa.

- **Origem**: Mesmo canto que a origem do espaço MACHINE
- **Propósito**: Define a área real onde os trabalhos podem ser executados
- **Margens**: Podem ter margens aplicadas (esquerda, superior, direita, inferior)

Por exemplo, se sua máquina tem 400×300mm mas tem uma margem de 10mm em todos os lados,
a WORKAREA seria 380×280mm começando na posição (10, 10) no espaço MACHINE.

## Entendendo WCS

Pense no WCS como "pontos zero" personalizáveis para seu trabalho. Enquanto sua máquina tem uma posição de origem fixa (determinada pelos finais de curso), o WCS permite que você defina onde quer que seu trabalho comece.

### Por Que Usar WCS?

- **Múltiplas fixações**: Configure várias áreas de trabalho na sua mesa e alterne entre elas
- **Posicionamento repetível**: Execute o mesmo trabalho em diferentes locais
- **Alinhamento rápido**: Defina um ponto de referência baseado no seu material ou peça
- **Fluxos de trabalho de produção**: Organize múltiplos trabalhos em sua área de trabalho

## Tipos de WCS

O Rayforge suporta os seguintes sistemas de coordenadas:

| Sistema | Tipo       | Descrição                                                         |
| ------- | ---------- | ----------------------------------------------------------------- |
| **G53** | Máquina    | Coordenadas absolutas de máquina (fixas, não podem ser alteradas) |
| **G54** | Trabalho 1 | Primeiro sistema de coordenadas de trabalho (padrão)              |
| **G55** | Trabalho 2 | Segundo sistema de coordenadas de trabalho                        |
| **G56** | Trabalho 3 | Terceiro sistema de coordenadas de trabalho                       |
| **G57** | Trabalho 4 | Quarto sistema de coordenadas de trabalho                         |
| **G58** | Trabalho 5 | Quinto sistema de coordenadas de trabalho                         |
| **G59** | Trabalho 6 | Sexto sistema de coordenadas de trabalho                          |

### Coordenadas de Máquina (G53)

G53 representa a posição absoluta da sua máquina, com zero na posição de origem da máquina. Isso é fixado pelo seu hardware e não pode ser alterado.

**Quando usar:**

- Homing e calibração
- Posicionamento absoluto relativo aos limites da máquina
- Quando você precisa referenciar a posição física da máquina

### Coordenadas de Trabalho (G54-G59)

Estes são sistemas de coordenadas deslocados que você pode definir. Cada um tem seu próprio ponto zero que você pode definir em qualquer lugar da sua área de trabalho.

**Quando usar:**

- Configurar múltiplas fixações de trabalho
- Alinhar a posições de material
- Executar o mesmo trabalho em diferentes locais

## Visualizando WCS na Interface

### Tela 2D

A tela 2D mostra sua origem WCS com um marcador verde:

- **Linhas verdes**: Indicam a posição atual da origem WCS (0, 0)
- **Alinhamento da grade**: As linhas da grade estão alinhadas à origem WCS, não à origem da máquina

O marcador de origem se move quando você altera o WCS ativo ou seu deslocamento, mostrando exatamente onde seu trabalho começará.

### Visualização 3D

Na visualização 3D, o WCS é exibido de forma diferente:

- **Grade e eixos**: Toda a grade aparece como se a origem WCS fosse a origem do mundo
- **Visão isolada**: O WCS é mostrado "em isolamento" - parece que a grade está centrada no WCS, não na máquina
- **Rótulos**: Os rótulos de coordenadas são relativos à origem WCS

Isso facilita visualizar onde seu trabalho será executado em relação ao sistema de coordenadas de trabalho selecionado.

## Selecionando e Alterando WCS

### Via Barra de Ferramentas

1. Localize o menu suspenso WCS na barra de ferramentas principal (rotulado "G53" por padrão)
2. Clique para ver os sistemas de coordenadas disponíveis
3. Selecione o WCS que deseja usar

### Via Painel de Controle

1. Abra o Painel de Controle (Visualizar → Painel de Controle ou Ctrl+L)
2. Encontre o menu suspenso WCS na seção de status da máquina
3. Selecione o WCS desejado no menu suspenso

## Definindo Deslocamentos WCS

Você pode definir onde cada origem WCS está localizada na sua máquina.

### Definindo Zero na Posição Atual

1. Conecte-se à sua máquina
2. Selecione o WCS que deseja configurar (ex., G54)
3. Mova a cabeça do laser para a posição que deseja que seja (0, 0)
4. No Painel de Controle, clique nos botões de zero:
   - **Zero X**: Define a posição X atual como 0 para o WCS ativo
   - **Zero Y**: Define a posição Y atual como 0 para o WCS ativo
   - **Zero Z**: Define a posição Z atual como 0 para o WCS ativo

Os deslocamentos são armazenados no controlador da sua máquina e persistem entre sessões.

### Vendo Deslocamentos Atuais

O Painel de Controle mostra os deslocamentos atuais para o WCS ativo:

- **Deslocamentos Atuais**: Exibe o deslocamento (X, Y, Z) da origem da máquina
- **Posição Atual**: Mostra a posição da cabeça do laser no WCS ativo

## WCS em seus Trabalhos

Quando você executa um trabalho, o Rayforge usa o WCS ativo para posicionar seu trabalho:

1. Projete seu trabalho na tela
2. Selecione o WCS que deseja usar
3. Execute o trabalho - ele será posicionado de acordo com o deslocamento WCS

O mesmo trabalho pode ser executado em diferentes posições simplesmente alterando o WCS ativo.

## Fluxos de Trabalho Práticos

### Fluxo de Trabalho 1: Múltiplas Posições de Fixação

Você tem uma mesa grande e quer configurar três áreas de trabalho:

1. **Faça homing na sua máquina** para estabelecer uma referência
2. **Mova para a primeira área de trabalho** e defina o deslocamento G54 (Zero X, Zero Y)
3. **Mova para a segunda área de trabalho** e defina o deslocamento G55
4. **Mova para a terceira área de trabalho** e defina o deslocamento G56
5. Agora você pode alternar entre G54, G55 e G56 para executar trabalhos em cada área

### Fluxo de Trabalho 2: Alinhando ao Material

Você tem uma peça de material colocada em algum lugar da sua mesa:

1. **Mova a cabeça do laser** para o canto do seu material
2. **Selecione G54** (ou seu WCS preferido)
3. **Clique em Zero X e Zero Y** para definir o canto do material como (0, 0)
4. **Projete seu trabalho** com (0, 0) como origem
5. **Execute o trabalho** - ele começará do canto do material

### Fluxo de Trabalho 3: Grade de Produção

Você precisa cortar a mesma peça 10 vezes em diferentes locais:

1. **Projete uma peça** no Rayforge
2. **Configure os deslocamentos G54-G59** para suas posições desejadas
3. **Execute o trabalho** com G54 ativo
4. **Mude para G55** e execute novamente
5. **Repita** para cada posição WCS

## Notas Importantes

### Limitações do WCS

- **G53 não pode ser alterado**: As coordenadas de máquina são fixadas pelo hardware
- **Os deslocamentos persistem**: Os deslocamentos WCS são armazenados no controlador da sua máquina
- **Conexão necessária**: Você deve estar conectado a uma máquina para definir deslocamentos WCS

---

**Páginas Relacionadas:**

- [Painel de Controle](../ui/control-panel) - Controle manual e gerenciamento do WCS
- [Configuração de Máquina](../machine/general) - Configure sua máquina
- [Visualização 3D](../ui/3d-preview) - Visualizando seus trabalhos
