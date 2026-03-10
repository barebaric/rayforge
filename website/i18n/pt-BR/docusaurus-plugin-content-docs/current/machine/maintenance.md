# Manutenção

A página de Manutenção nas Configurações da Máquina ajuda você a rastrear o uso da máquina e agendar tarefas de manutenção.

![Configurações de Manutenção](/screenshots/machine-maintenance.png)

## Rastreamento de Uso

O Rayforge rastreia quanto tempo sua máquina está em uso. Essas informações ajudam você a agendar manutenção preventiva em intervalos apropriados.

### Horas Totais

O contador de horas totais rastreia todo o tempo gasto executando trabalhos na máquina. Este contador cumulativo não pode ser resetado e fornece um histórico completo do uso da máquina.

Use isso para rastrear a idade geral da máquina e planejar intervalos de serviço principais.

## Contadores de Manutenção Personalizados

Você pode criar contadores personalizados para rastrear intervalos de manutenção específicos. Cada contador tem um nome, rastreia horas e pode ser configurado com um limite de notificação.

### Criando um Contador

1. Clique no botão adicionar para criar um novo contador
2. Digite um nome descritivo (ex: "Tubo Laser", "Tensão de Correia", "Limpeza de Espelho")
3. Defina um limite de notificação em horas se desejado

### Recursos do Contador

- **Nomes personalizados**: Rotule contadores para qualquer tarefa de manutenção
- **Rastreamento de horas**: Acumula automaticamente tempo durante a execução de trabalhos
- **Limites de notificação**: Receba lembretes quando manutenção é necessária
- **Capacidade de reset**: Reset contadores após realizar manutenção

### Exemplos de Contadores

**Tubo Laser**: Rastreie horas do tubo CO2 para planejar substituição (tipicamente 1000-3000 horas). Defina uma notificação às 2500 horas para planejar com antecedência.

**Tensão de Correia**: Rastreie horas desde a última tensão de correia. Reset após realizar a manutenção.

**Limpeza de Espelho**: Rastreie uso desde a última limpeza de espelho. Reset após limpar.

**Lubrificação de Rolamentos**: Rastreie horas para intervalos de manutenção de rolamentos.

## Resetando Contadores

Após realizar manutenção, você pode resetar o contador relevante:

1. Clique no botão de reset ao lado do contador
2. Confirme o reset no diálogo
3. O contador retorna para zero

:::tip Cronograma de Manutenção
Intervalos comuns de manutenção:
- **Diariamente**: Limpar lente, verificar alinhamento do espelho
- **Semanalmente**: Limpar trilhos, verificar tensão das correias
- **Mensalmente**: Lubrificar rolamentos, verificar conexões elétricas
- **Anualmente**: Inspeção completa, substituir peças gastas

Ajuste intervalos com base em seus padrões de uso e recomendações do fabricante.
:::

## Veja Também

- [Configurações do Laser](laser) - Configuração da cabeça do laser
- [Configurações de Hardware](hardware) - Dimensões da máquina
