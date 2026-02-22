# Modo de Simulação

![Modo de Simulação](/screenshots/main-simulation.png)

O Modo de Simulação fornece visualização em tempo real da execução do seu trabalho a laser antes de executá-lo na máquina real. Mostra a ordem de execução, variações de velocidade e níveis de potência através de uma sobreposição interativa na visualização 2D.

## Visão Geral

O Modo de Simulação ajuda você a:

- **Visualizar a ordem de execução** - Veja a sequência exata que as operações vão rodar
- **Identificar variações de velocidade** - Mapa de calor colorido mostra movimentos lentos (azul) para rápidos (vermelho)
- **Verificar níveis de potência** - Transparência indica potência (fraco=baixa, forte=alta)
- **Validar testes de material** - Confirme a ordem de execução da grade de teste
- **Detectar erros cedo** - Identifique problemas antes de desperdiçar material
- **Entender tempo** - Veja quanto tempo diferentes operações levam


## Ativando o Modo de Simulação

Existem três formas de entrar no Modo de Simulação:

### Método 1: Atalho de Teclado
Pressione <kbd>f7</kbd> para alternar o modo de simulação ligado/desligado.

### Método 2: Menu
- Navegue até **Visualizar → Simular Execução**
- Clique para alternar ligado/desligado

### Método 3: Barra de Ferramentas (se disponível)
- Clique no botão do modo de simulação na barra de ferramentas

:::note Apenas Visualização 2D
O modo de simulação funciona na visualização 2D. Se você está na visualização 3D (<kbd>f6</kbd>), mude para a visualização 2D (<kbd>f5</kbd>) primeiro.
:::


## Entendendo a Visualização

### Mapa de Calor de Velocidade

As operações são coloridas com base em sua velocidade:

| Cor | Velocidade | Significado |
|-------|-------|---------|
| 🔵 **Azul** | Mais lenta | Velocidade mínima no seu trabalho |
| 🔵 **Ciano** | Lenta | Abaixo da velocidade média |
| 🟢 **Verde** | Média | Velocidade média |
| 🟡 **Amarelo** | Rápida | Acima da velocidade média |
| 🔴 **Vermelho** | Mais rápida | Velocidade máxima no seu trabalho |

O mapa de calor é **normalizado** para o intervalo real de velocidades do seu trabalho:
- Se seu trabalho roda a 100-1000 mm/min, azul=100, vermelho=1000
- Se seu trabalho roda a 5000-10000 mm/min, azul=5000, vermelho=10000


### Transparência de Potência

A opacidade da linha indica a potência do laser:

- **Linhas fracas** (10% opacidade) = Baixa potência (0%)
- **Translúcidas** (50% opacidade) = Potência média (50%)
- **Linhas sólidas** (100% opacidade) = Potência máxima (100%)

Isso ajuda a identificar:
- Movimentos de deslocamento (0% potência) - Muito fracos
- Operações de gravação - Opacidade moderada
- Operações de corte - Linhas sólidas e fortes

### Indicador da Cabeça do Laser

A posição do laser é mostrada com uma mira:

- 🔴 Mira vermelha (linhas de 6mm)
- Contorno circular (raio de 3mm)
- Ponto central (0.5mm)

O indicador se move durante a reprodução, mostrando exatamente onde o laser está na sequência de execução.

## Controles de Reprodução

Quando o modo de simulação está ativo, controles de reprodução aparecem na parte inferior da tela:


### Botão Reproduzir/Pausar

- **▶️ Reproduzir**: Inicia a reprodução automática
- **⏸️ Pausar**: Para na posição atual
- **Auto-reprodução**: A reprodução inicia automaticamente quando você ativa o modo de simulação

### Controle Deslizante de Progresso

- **Arraste** para percorrer a execução
- **Clique** para pular para um ponto específico
- Mostra passo atual / total de passos
- Suporta posições fracionárias para percorrer suavemente

### Exibição do Intervalo de Velocidade

Mostra as velocidades mínima e máxima no seu trabalho:

```
Intervalo de velocidade: 100 - 5000 mm/min
```

Isso ajuda a entender as cores do mapa de calor.

## Usando o Modo de Simulação

### Validando a Ordem de Execução

A simulação mostra a ordem exata que as operações vão executar:

1. Ative o modo de simulação (<kbd>f7</kbd>)
2. Assista a reprodução
3. Verifique se as operações rodam na sequência esperada
4. Confirme que cortes acontecem após gravação (se aplicável)

**Exemplo:** Grade de teste de material
- Observe a ordem otimizada por risco (velocidades mais rápidas primeiro)
- Confirme que células de baixa potência executam antes de alta potência
- Valide que o teste roda em sequência segura

### Verificando Variações de Velocidade

Use o mapa de calor para identificar mudanças de velocidade:

- **Cor consistente** = Velocidade uniforme (bom para gravação)
- **Mudanças de cor** = Variações de velocidade (esperado em cantos)
- **Áreas azuis** = Movimentos lentos (verifique se é intencional)

### Estimando Tempo do Trabalho

A duração da reprodução é escalada para 5 segundos para o trabalho completo:

- Assista a velocidade de reprodução
- Estime o tempo real: Se a reprodução parece suave, o trabalho será rápido
- Se a reprodução pula rapidamente, o trabalho tem muitos segmentos pequenos

:::tip Tempo Real
 Para o tempo real do trabalho durante execução (não simulação), verifique a seção
 direita da barra de status após gerar o G-code.
 :::


### Depurando Testes de Material

Para grades de teste de material, a simulação mostra:

1. **Ordem de execução** - Verifique se as células rodam mais rápido→mais lento
2. **Mapa de calor de velocidade** - Cada coluna deve ter uma cor diferente
3. **Transparência de potência** - Cada linha deve ter opacidade diferente

Isso ajuda a confirmar que o teste vai rodar corretamente antes de usar material.

## Editando Durante a Simulação

Diferente de muitas ferramentas CAM, o Rayforge permite **editar peças de trabalho durante a simulação**:

- Mover, escalar, rotacionar objetos ✅
- Mudar configurações de operação ✅
- Adicionar/remover peças de trabalho ✅
- Zoom e panorâmica ✅

**Atualização automática:** A simulação atualiza automaticamente quando você muda as configurações.

:::note Sem Troca de Contexto
Você pode permanecer no modo de simulação enquanto edita - não precisa alternar entre um e outro.
:::


## Dicas e Melhores Práticas

### Quando Usar Simulação

✅ **Sempre simule antes de:**
- Rodar materiais caros
- Trabalhos longos (>30 minutos)
- Grades de teste de material
- Trabalhos com ordens de execução complexas

✅ **Use simulação para:**
- Verificar ordem de operações
- Verificar movimentos de deslocamento inesperados
- Validar configurações de velocidade/potência
- Treinar novos usuários

### Lendo a Visualização

✅ **Procure:**
- Cores consistentes dentro de operações (bom)
- Transições suaves entre segmentos (bom)
- Áreas azuis inesperadas (investigue - por que tão lento?)
- Linhas fracas em áreas de corte (errado - verifique configurações de potência)

⚠️ **Bandeiras vermelhas:**
- Cortar antes de gravar (a peça de trabalho pode se mover)
- Seções azuis (lentas) muito longas (ineficiente)
- Mudanças de potência no meio da operação (verifique configurações)

### Dicas de Desempenho

- A simulação atualiza automaticamente nas mudanças
- Para trabalhos muito complexos (1000+ operações), a simulação pode ficar lenta
- Desative a simulação (<kbd>f7</kbd>) quando não precisar para melhor desempenho

## Atalhos de Teclado

| Atalho | Ação |
|----------|--------|
| <kbd>f7</kbd> | Alternar modo de simulação ligado/desligado |
| <kbd>f5</kbd> | Mudar para visualização 2D (necessário para simulação) |
| <kbd>espaço</kbd> | Reproduzir/Pausar reprodução |
| <kbd>esquerda</kbd> | Retroceder um passo |
| <kbd>direita</kbd> | Avançar um passo |
| <kbd>home</kbd> | Pular para o início |
| <kbd>end</kbd> | Pular para o final |

## Tópicos Relacionados

- **[Visualização 3D](../ui/3d-preview)** - Visualização 3D do caminho da ferramenta
- **[Grade de Teste de Material](operations/material-test-grid)** - Use simulação para validar testes
- **[Simulando Seu Trabalho](simulating-your-job)** - Guia detalhado de simulação
