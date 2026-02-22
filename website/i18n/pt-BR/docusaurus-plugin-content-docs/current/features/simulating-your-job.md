# Simulando Seu Trabalho

![Captura de Tela do Modo de Simulação](/screenshots/main-simulation.png)

Aprenda a usar o modo de simulação do Rayforge para visualizar seu trabalho a laser, identificar possíveis problemas e estimar o tempo de conclusão antes de executar no hardware real.

## Visão Geral

O modo de simulação permite visualizar a execução do seu trabalho a laser sem realmente rodar a máquina. Isso ajuda a detectar erros, otimizar configurações e planejar seu fluxo de trabalho.

## Benefícios da Simulação

- **Visualizar execução do trabalho**: Veja exatamente como o laser vai se mover
- **Estimar tempo**: Obtenha estimativas precisas de duração do trabalho
- **Identificar problemas**: Detecte sobreposições, lacunas ou comportamento inesperado
- **Otimizar ordem do caminho**: Visualize a sequência de corte
- **Aprender G-code**: Entenda como as operações se traduzem em comandos da máquina

## Iniciando uma Simulação

1. **Carregue ou crie seu design** no Rayforge
2. **Configure as operações** com as configurações desejadas
3. **Clique no botão Simular** na barra de ferramentas (ou use o atalho de teclado)
4. **Assista a simulação** percorrer seu trabalho

## Controles de Simulação

### Controles de Reprodução

- **Reproduzir/Pausar**: Inicia ou pausa a simulação
- **Avançar/Voltar**: Percorre o trabalho um comando por vez
- **Controle de Velocidade**: Ajusta a velocidade de reprodução (0.5x a 10x)
- **Pular para Posição**: Vai para uma porcentagem específica do trabalho
- **Reiniciar**: Inicia a simulação do começo

### Opções de Visualização

- **Mostrar caminho da ferramenta**: Exibe o caminho que a cabeça do laser vai seguir
- **Mostrar movimentos de deslocamento**: Visualiza movimentos rápidos de posicionamento
- **Mostrar potência do laser**: Colore caminhos por nível de potência
- **Modo mapa de calor**: Visualiza tempo de permanência e densidade de potência

### Exibição de Informações

Durante a simulação, monitore:

- **Posição atual**: Coordenadas X, Y da cabeça do laser
- **Progresso do trabalho**: Porcentagem concluída
- **Tempo estimado restante**: Baseado no progresso atual
- **Operação atual**: Qual operação está sendo executada
- **Potência e velocidade**: Parâmetros atuais do laser

## Interpretando Resultados da Simulação

### O Que Procurar

- **Eficiência do caminho**: Existem movimentos de deslocamento desnecessários?
- **Cortes sobrepostos**: Corte duplo não intencional de caminhos
- **Ordem das operações**: A sequência faz sentido?
- **Distribuição de potência**: A potência está sendo aplicada consistentemente?
- **Movimentos inesperados**: Qualquer padrão de movimento brusco ou estranho

### Visualização de Mapa de Calor

O mapa de calor mostra exposição cumulativa do laser:

- **Cores frias (azul/verde)**: Baixa exposição
- **Cores quentes (amarelo/laranja)**: Exposição moderada
- **Cores quentes (vermelho)**: Alta exposição ou tempo de permanência

Use isso para identificar:

- **Pontos quentes**: Áreas que podem queimar demais
- **Lacunas**: Áreas que podem estar subexpostas
- **Problemas de sobreposição**: Exposição dupla não intencional

Veja [Modo de Simulação](../features/simulation-mode) para informações detalhadas.

## Usando Simulação para Otimização

### Otimizar Ordem de Corte

Se a simulação revelar ordem de caminho ineficiente:

1. **Ative otimização de caminho** nas configurações da operação
2. **Escolha o método de otimização** (vizinho mais próximo, TSP)
3. **Re-simule** para verificar a melhoria

### Ajustar Tempo

A simulação fornece estimativas de tempo precisas:

- **Tempos de trabalho longos**: Considere otimizar caminhos ou aumentar velocidade
- **Tempos muito curtos**: Verifique se as configurações estão corretas para o material
- **Duração inesperada**: Verifique operações ocultas ou duplicadas

### Verificar Trabalhos Multi-Camadas

Para projetos complexos multi-camadas:

1. **Simule cada camada** independentemente
2. **Verifique a ordem das operações** entre camadas
3. **Verifique conflitos** entre camadas
4. **Estime o tempo total** para o trabalho completo

## Simulação vs. Execução Real

### Diferenças a Observar

A simulação é altamente precisa mas:

- **Não considera**: Imperfeições mecânicas, backlash, vibração
- **Pode diferir ligeiramente**: Aceleração/desaceleração real vs. simulada
- **Não mostra**: Interação com material, fumaça, vapores
- **Estimativas de tempo**: Geralmente precisas dentro de 5-10%

### Quando Re-simular

- **Após mudar configurações**: Potência, velocidade ou parâmetros de operação
- **Após editar o design**: Quaisquer mudanças de design
- **Antes de materiais caros**: Verifique novamente antes de comprometer
- **Ao solucionar problemas**: Verifique correções para problemas identificados

## Dicas para Simulação Eficaz

- **Sempre simule** antes de executar trabalhos importantes
- **Use reprodução mais lenta** para detectar problemas sutis
- **Ative mapa de calor** para trabalhos de gravação
- **Compare múltiplas configurações** simulando variações
- **Documente resultados**: Capture tela ou anote problemas encontrados

## Solução de Problemas da Simulação

**Simulação não inicia**: Verifique se as operações estão configuradas corretamente

**Simulação roda muito rápido**: Ajuste a velocidade de reprodução para uma configuração mais lenta

**Não consegue ver detalhes**: Aplique zoom em áreas específicas de interesse

**Estimativa de tempo parece errada**: Verifique se o perfil da máquina tem velocidades máximas corretas

## Tópicos Relacionados

- [Recurso Modo de Simulação](../features/simulation-mode)
- [Fluxo de Trabalho Multi-Camadas](../features/multi-layer)
