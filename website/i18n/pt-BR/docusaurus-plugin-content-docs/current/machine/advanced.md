# Configurações Avançadas

A página Avançado nas Configurações da Máquina contém opções de configuração adicionais para casos
de uso especializados.

![Configurações Avançadas](/screenshots/machine-settings-advanced.webp)

## Comportamento de Conexão

Configurações que controlam como o Rayforge interage com sua máquina durante a conexão.

### Origem ao Conectar

Quando habilitado, o Rayforge envia automaticamente um comando de homing ($H) ao conectar à máquina.

- **Habilite se**: Sua máquina tem chaves de limite confiáveis
- **Desabilite se**: Sua máquina não tem chaves de limite ou homing não é confiável

### Limpar Alarmes ao Conectar

Quando habilitado, o Rayforge limpa automaticamente qualquer estado de alarme ao conectar.

- **Habilite se**: Sua máquina frequentemente inicia em estado de alarme
- **Desabilite se**: Você quer investigar alarmes manualmente antes de limpar

### Permitir Homing de Eixo Único

Quando habilitado, você pode fazer homing de eixos individuais independentemente (X, Y ou Z) em vez
de requerer todos os eixos juntos. Isso é útil para máquinas onde um eixo já pode estar posicionado
corretamente.

## Configurações de Arcos e Curvas

Configurações para controlar como caminhos curvos são convertidos em movimentos de G-code.

### Suportar Arcos

Quando habilitado, o Rayforge gera comandos de arco (G2/G3) para caminhos curvos em vez de
dividi-los em muitos movimentos lineares pequenos. Isso produz G-code mais compacto e movimento mais
suave na maioria dos controladores.

Quando desabilitado, todas as curvas são convertidas em segmentos lineares (comandos G1), o que
fornece compatibilidade máxima com controladores que não suportam arcos.

### Suportar Curvas Bézier

Quando habilitado, o Rayforge gera comandos Bézier cúbicos nativos (como o comando G5 usado pelo
LinuxCNC) para caminhos curvos. Isso produz movimento muito suave e G-code compacto em controladores
que o suportam. Você deve desabilitar esta configuração se o firmware da sua máquina não entender
comandos Bézier, caso em que as curvas serão divididas em segmentos lineares.

### Tolerância de Arco e Curva

Esta configuração controla o desvio máximo permitido ao ajustar arcos e curvas a caminhos curvos,
especificado em milímetros. Um valor menor produz caminhos mais precisos mas pode requerer mais
comandos. Um valor maior permite mais desvio mas gera menos comandos.

Valores típicos variam de 0.01mm para trabalho de precisão a 0.1mm para processamento mais rápido.

## Deslocamento de Varredura Bidirecional por Velocidade {#bidirectional-scan-offset-by-speed}

A tabela de deslocamento de varredura bidirecional por velocidade corrige o desalinhamento
horizontal entre passagens raster da esquerda para a direita e da direita para a esquerda. Em muitas
máquinas esse desalinhamento não é constante: quanto mais rápido o cabeçote do laser se move, mais
as passagens da direita para a esquerda saem do alinhamento, o que aparece como faixas que pioram em
velocidades de gravação mais altas.

![Deslocamento de Varredura Bidirecional por Velocidade](/screenshots/machine-settings-advanced-bidir-offset-table.webp)

Cada linha da tabela associa uma velocidade de gravação ao deslocamento, em milímetros, de que as
passagens da direita para a esquerda precisam nessa velocidade:

- Use **Adicionar Linha** para adicionar uma linha e o botão de lixeira de uma linha para removê-la
- Valores positivos deslocam as passagens da direita para a esquerda para a direita; valores
  negativos, para a esquerda
- As linhas são mantidas ordenadas por velocidade; se duas linhas tiverem a mesma velocidade, a
  última prevalece

Quando uma etapa de gravação é executada, o Rayforge consulta a velocidade da etapa nesta tabela e
interpola o deslocamento linearmente entre as linhas vizinhas. Velocidades abaixo da primeira linha
ou acima da última usam o deslocamento da linha mais próxima, e uma tabela vazia significa que todas
as velocidades usam um deslocamento de 0.

**Sem correção** — o desalinhamento cresce com a velocidade de gravação:

![Passagens raster desalinhadas a 1000, 4000 e 8000 mm/min](/images/bidir-scan-offset-uncorrected.webp)

**Com a tabela de deslocamento aplicada:**

![Passagens raster alinhadas em todas as velocidades](/images/bidir-scan-offset-table-applied.webp)

A tabela se aplica apenas a etapas de gravação cujo próprio **Deslocamento de Varredura
Bidirecional** é 0; um deslocamento por etapa diferente de zero sempre tem precedência. Consulte
[Deslocamento de Varredura Bidirecional](../features/operations/engrave.md#bidirectional-scan-offset)
para ver como fica o desalinhamento e como calibrá-lo.

## Veja Também

- [Configurações de Hardware](hardware) - Configuração de origem dos eixos e inversão
- [Configurações do Dispositivo](device) - Configurações específicas de GRBL
