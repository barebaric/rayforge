# Sistemas de Coordenadas de Trabalho (WCS)

Sistemas de Coordenadas de Trabalho (WCS) permitem definir múltiplos pontos de referência na
área de trabalho da sua máquina. Isso facilita executar o mesmo trabalho em
posições diferentes sem redesenhar ou reposicionar suas peças de trabalho.

## Entendendo WCS

Pense no WCS como "pontos zero" personalizáveis para seu trabalho. Enquanto sua
máquina tem uma posição home fixa (determinada pelos switches de limite), o WCS permite
definir onde você quer que seu trabalho comece.

### Por Que Usar WCS?

- **Múltiplas fixações**: Configure várias áreas de trabalho na sua mesa e alterne
  entre elas
- **Posicionamento repetível**: Execute o mesmo trabalho em locais diferentes
- **Alinhamento rápido**: Defina um ponto de referência baseado no seu material ou
  peça de trabalho
- **Fluxos de produção**: Organize múltiplos trabalhos pela sua área de trabalho

## Tipos de WCS

O Rayforge suporta os seguintes sistemas de coordenadas:

| Sistema  | Tipo    | Descrição                                             |
| ------- | ------- | ------------------------------------------------------- |
| **G53** | Máquina | Coordenadas absolutas da máquina (fixo, não pode ser alterado) |
| **G54** | Trabalho 1  | Primeiro sistema de coordenadas de trabalho (padrão)                  |
| **G55** | Trabalho 2  | Segundo sistema de coordenadas de trabalho                           |
| **G56** | Trabalho 3  | Terceiro sistema de coordenadas de trabalho                            |
| **G57** | Trabalho 4  | Quarto sistema de coordenadas de trabalho                           |
| **G58** | Trabalho 5  | Quinto sistema de coordenadas de trabalho                            |
| **G59** | Trabalho 6  | Sexto sistema de coordenadas de trabalho                            |

### Coordenadas de Máquina (G53)

G53 representa a posição absoluta da sua máquina, com zero na
posição home da máquina. Isso é fixado pelo seu hardware e não pode ser
alterado.

**Quando usar:**

- Homing e calibração
- Posicionamento absoluto relativo aos limites da máquina
- Quando você precisa referenciar a posição física da máquina

### Coordenadas de Trabalho (G54-G59)

Esses são sistemas de coordenadas com deslocamento que você pode definir. Cada um tem seu próprio
ponto zero que você pode definir em qualquer lugar da sua área de trabalho.

**Quando usar:**

- Configurar múltiplas fixações de trabalho
- Alinhar a posições de material
- Executar o mesmo trabalho em locais diferentes

## Visualizando WCS na Interface

### Tela 2D

A tela 2D mostra sua origem WCS com um marcador verde:

- **Linhas verdes**: Indicam a posição da origem WCS atual (0, 0)
- **Alinhamento da grade**: Linhas da grade estão alinhadas à origem WCS, não à origem
  da máquina

O marcador de origem se move quando você muda o WCS ativo ou seu deslocamento,
mostrando exatamente onde seu trabalho vai começar.

### Visualização 3D

Na visualização 3D, o WCS é exibido de forma diferente:

- **Grade e eixos**: A grade inteira aparece como se a origem WCS fosse a
  origem do mundo
- **Visualização isolada**: O WCS é mostrado "em isolamento" - parece que a
  grade está centrada no WCS, não na máquina
- **Rótulos**: Rótulos de coordenadas são relativos à origem WCS

Isso facilita visualizar onde seu trabalho vai rodar em relação ao
sistema de coordenadas de trabalho selecionado.

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

1. Conecte à sua máquina
2. Selecione o WCS que deseja configurar (ex., G54)
3. Mova a cabeça do laser para a posição que você quer que seja (0, 0)
4. No Painel de Controle, clique nos botões de zero:
   - **Zero X**: Define a posição X atual como 0 para o WCS ativo
   - **Zero Y**: Define a posição Y atual como 0 para o WCS ativo
   - **Zero Z**: Define a posição Z atual como 0 para o WCS ativo

Os deslocamentos são armazenados no controlador da sua máquina e persistem entre
sessões.

### Visualizando Deslocamentos Atuais

O Painel de Controle mostra os deslocamentos atuais para o WCS ativo:

- **Deslocamentos Atuais**: Exibe o deslocamento (X, Y, Z) da origem da máquina
- **Posição Atual**: Mostra a posição da cabeça do laser no WCS ativo

## WCS nos Seus Trabalhos

Quando você executa um trabalho, o Rayforge usa o WCS ativo para posicionar seu trabalho:

1. Projete seu trabalho na tela
2. Selecione o WCS que deseja usar
3. Execute o trabalho - ele será posicionado de acordo com o deslocamento WCS

O mesmo trabalho pode ser executado em posições diferentes simplesmente mudando o
WCS ativo.

## Fluxos de Trabalho Práticos

### Fluxo de Trabalho 1: Múltiplas Posições de Fixação

Você tem uma mesa grande e quer configurar três áreas de trabalho:

1. **Faça home da sua máquina** para estabelecer uma referência
2. **Mova para a primeira área de trabalho** e defina o deslocamento G54 (Zero X, Zero Y)
3. **Mova para a segunda área de trabalho** e defina o deslocamento G55
4. **Mova para a terceira área de trabalho** e defina o deslocamento G56
5. Agora você pode alternar entre G54, G55 e G56 para executar trabalhos em cada área

### Fluxo de Trabalho 2: Alinhando ao Material

Você tem uma peça de material colocada em algum lugar da sua mesa:

1. **Mova a cabeça do laser** para o canto do seu material
2. **Selecione G54** (ou seu WCS preferido)
3. **Clique Zero X e Zero Y** para definir o canto do material como (0, 0)
4. **Projete seu trabalho** com (0, 0) como origem
5. **Execute o trabalho** - ele vai começar do canto do material

### Fluxo de Trabalho 3: Grade de Produção

Você precisa cortar a mesma peça 10 vezes em locais diferentes:

1. **Projete uma peça** no Rayforge
2. **Configure deslocamentos G54-G59** para suas posições desejadas
3. **Execute o trabalho** com G54 ativo
4. **Mude para G55** e execute novamente
5. **Repita** para cada posição WCS

## Notas Importantes

### Limitações do WCS

- **G53 não pode ser alterado**: Coordenadas de máquina são fixadas pelo hardware
- **Deslocamentos persistem**: Deslocamentos WCS são armazenados no controlador da sua máquina
- **Conexão necessária**: Você deve estar conectado a uma máquina para definir deslocamentos
  WCS

### WCS e Origem do Trabalho

WCS funciona independentemente das suas configurações de origem do trabalho. A origem do trabalho determina
onde na tela seu trabalho começa, enquanto WCS determina onde essa
posição da tela mapeia na sua máquina.

### Compatibilidade de Máquina

Nem todas as máquinas suportam todos os recursos WCS:

- **GRBL (v1.1+)**: Suporte completo a G53-G59
- **Smoothieware**: Suporta G54-G59 (leitura de deslocamento pode ser limitada)
- **Controladores personalizados**: Varia por implementação

---

**Páginas Relacionadas:**

- [Sistemas de Coordenadas](coordinate-systems) - Entendendo sistemas de coordenadas
- [Painel de Controle](../ui/control-panel) - Controle manual e gerenciamento WCS
- [Configuração de Máquina](../machine/general) - Configure sua máquina
- [Visualização 3D](../ui/3d-preview) - Visualizando seus trabalhos
