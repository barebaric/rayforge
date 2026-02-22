# Máquinas

![Configurações de Máquinas](/screenshots/application-machines.png)

A página Máquinas nas Configurações da Aplicação permite gerenciar perfis
de máquina. Cada perfil contém toda a configuração para uma máquina
a laser específica.

## Perfis de Máquina

Perfis de máquina armazenam configuração completa para um cortador ou
gravador a laser, incluindo:

- **Configurações gerais**: Nome, velocidades, aceleração
- **Configurações de hardware**: Dimensões da área de trabalho, configuração de eixos
- **Configurações do laser**: Intervalo de potência, frequência PWM
- **Configurações do dispositivo**: Porta serial, taxa de transmissão, tipo de firmware
- **Configurações de G-code**: Opções de dialeto G-code personalizado
- **Configurações de câmera**: Calibração e alinhamento da câmera

## Gerenciando Máquinas

### Adicionando uma Nova Máquina

1. Clique no botão **Adicionar Nova Máquina**
2. Digite um nome descritivo para sua máquina
3. Configure as definições da máquina (veja
   [Configuração de Máquina](../machine/general) para detalhes)
4. Clique em **Salvar** para criar o perfil

### Alternando Entre Máquinas

Use o menu suspenso seletor de máquina na janela principal para alternar entre
máquinas configuradas. Todas as configurações, incluindo a máquina selecionada, são
lembradas entre sessões.

### Duplicando uma Máquina

Para criar um perfil de máquina similar:

1. Selecione a máquina a duplicar
2. Clique no botão **Duplicar**
3. Renomeie a nova máquina e ajuste as configurações conforme necessário

### Excluindo uma Máquina

1. Selecione a máquina a excluir
2. Clique no botão **Excluir**
3. Confirme a exclusão

:::warning
Excluir um perfil de máquina não pode ser desfeito. Certifique-se de ter
anotado quaisquer configurações importantes antes de excluir.
:::

## Tópicos Relacionados

- [Configuração de Máquina](../machine/general) - Configuração detalhada da máquina
- [Configuração Inicial](../getting-started/first-time-setup) - Guia de
  configuração inicial
