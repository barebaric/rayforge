---
slug: multi-laser-support
title: Suporte Multi-Laser - Escolha Diferentes Lasers para Cada Operação
authors: rayforge_team
tags: [multi-laser, operations, workflow]
---

![Sobreposição de Câmera](/images/camera-overlay.png)

Um dos recursos mais poderosos no Rayforge é a capacidade de atribuir
diferentes lasers a diferentes operações dentro de um único trabalho. Isso
abre possibilidades emocionantes para configurações multi-ferramentas e
fluxos de trabalho especializados.

<!-- truncate -->

## O que é Suporte Multi-Laser?

Se sua máquina está equipada com múltiplos módulos de laser — digamos, um
laser de diodo para gravação e um laser CO2 para corte, ou lasers de diodo
de diferentes potências otimizados para diferentes materiais — o Rayforge
permite aproveitar ao máximo essa configuração.

Com suporte multi-laser, você pode:

- **Atribuir diferentes lasers a diferentes operações** em seu trabalho
- **Alternar entre módulos de laser** automaticamente durante a execução
  do trabalho
- **Otimizar para material e tarefa** usando a ferramenta certa para cada
  operação

## Casos de Uso

### Gravação e Corte Híbrido

Imagine que você está trabalhando em um projeto de placa de madeira:

1. **Operação 1**: Use um laser de diodo de baixa potência para gravar
   texto fino e gráficos detalhados
2. **Operação 2**: Mude para um laser CO2 de maior potência para cortar
   o formato da placa

Com o Rayforge, você simplesmente atribui cada operação ao laser apropriado
em seu perfil de máquina, e o software cuida do resto.

### Otimização Específica por Material

Diferentes tipos de laser se destacam em diferentes materiais:

- **Lasers de diodo**: Ótimos para gravação em madeira, couro e alguns
  plásticos
- **Lasers CO2**: Excelentes para cortar acrílico, madeira e outros
  materiais orgânicos
- **Lasers de fibra**: Perfeitos para marcar metais

Se você tem múltiplos tipos de laser em um sistema de pórtico, o suporte
multi-laser do Rayforge permite usar a ferramenta ideal para cada parte do
seu projeto.

## Como Configurar

### 1. Configure Múltiplos Lasers em Seu Perfil de Máquina

Vá em **Configuração da Máquina → Múltiplos Lasers** e defina cada módulo
de laser em seu sistema. Você pode especificar:

- Tipo de laser e faixa de potência
- Posições de deslocamento (se os lasers estão montados em locais diferentes)
- Compatibilidade com materiais

Consulte nosso [Guia de Configuração de Laser](/docs/machine/laser)
para instruções detalhadas.

### 2. Atribua Lasers às Operações

Ao criar operações em seu projeto:

1. Selecione a operação (Contorno, Raster, etc.)
2. Nas configurações da operação, escolha qual laser usar no menu suspenso
3. Configure os parâmetros da operação específicos para aquele laser

### 3. Visualize e Execute

Use a visualização 3D para verificar suas trajetórias de ferramenta, depois
envie o trabalho para sua máquina. O Rayforge gerará automaticamente os
comandos G-code apropriados para alternar entre lasers conforme necessário.

## Detalhes Técnicos

Nos bastidores, o Rayforge usa comandos G-code para alternar entre módulos
de laser. A implementação exata depende do seu firmware e configuração de
hardware, mas abordagens comuns incluem:

- **M3/M4 com deslocamentos de ferramenta**: Alterna entre lasers usando
  comandos de troca de ferramenta
- **Controle GPIO**: Usa pinos GPIO suportados pelo firmware para
  habilitar/desabilitar diferentes módulos de laser
- **Macros personalizados**: Define macros pré e pós-operação para troca
  de laser

## Começando

O suporte multi-laser está disponível no Rayforge 0.15 e posterior. Para
começar:

1. Atualize para a versão mais recente
2. Configure seu perfil de máquina com múltiplos lasers
3. Experimente em um projeto de teste!

Confira a [documentação de Perfis de Máquina](/docs/machine/general)
para mais detalhes.

---

*Tem uma configuração multi-laser? Adoraríamos ouvir sobre sua experiência!
Compartilhe seus projetos e feedback no
[GitHub](https://github.com/barebaric/rayforge).*
