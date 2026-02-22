# Enquadrando Seu Trabalho

Aprenda a usar o recurso de enquadramento para visualizar os limites do seu trabalho e garantir o alinhamento correto antes de cortar.

## Visão Geral

O enquadramento permite visualizar os limites exatos do seu trabalho a laser traçando um contorno com o laser em baixa potência ou com o laser desligado. Isso ajuda a verificar o posicionamento e prevenir erros custosos.

## Quando Usar Enquadramento

- **Configurações iniciais**: Verifique o posicionamento do material
- **Posicionamento preciso**: Garanta que o design caiba dentro dos limites do material
- **Múltiplos trabalhos**: Confirme o alinhamento antes de cada execução
- **Materiais caros**: Verifique novamente antes de comprometer com cortes

## Como Enquadrar

### Método 1: Apenas Contorno

Trace o limite do trabalho sem ligar o laser:

1. **Carregue seu design** no Rayforge
2. **Posicione o material** na mesa do laser
3. **Clique no botão Enquadrar** na barra de ferramentas
4. **Observe a cabeça do laser** traçar o retângulo delimitador
5. **Verifique o posicionamento** e ajuste o material se necessário

### Método 2: Visualização em Baixa Potência

Algumas máquinas suportam enquadramento em baixa potência com um feixe visível:

1. **Ative o modo de baixa potência** nas configurações da máquina
2. **Defina a potência de enquadramento** (tipicamente 1-5%)
3. **Execute a operação de enquadramento**
4. **Observe o contorno** traçado na superfície do material

:::warning Verifique Sua Máquina
Nem todos os lasers suportam enquadramento em baixa potência com segurança. Consulte a documentação da sua máquina antes de usar este recurso.
:::


## Configurações de Enquadramento

Configure o comportamento do enquadramento em Configurações → Máquina:

- **Velocidade de enquadramento**: Quão rápido a cabeça do laser se move durante o enquadramento
- **Potência de enquadramento**: Potência do laser durante enquadramento (0 para desligado, baixa % para traço visível)
- **Pausa nos cantos**: Breve pausa em cada canto para visibilidade
- **Contagem de repetições**: Número de vezes para traçar o contorno

## Usando Resultados do Enquadramento

Após o enquadramento, você pode:

- **Ajustar a posição do material** se necessário
- **Re-enquadrar** para verificar a nova posição
- **Prosseguir com o trabalho** uma vez satisfeito com o posicionamento

## Dicas para Enquadramento Eficaz

- **Marque os cantos**: Coloque pequenos pedaços de fita nos cantos para referência
- **Verifique o espaçamento**: Garanta espaço adequado ao redor do seu design
- **Verifique a orientação**: Confirme se o material está orientado corretamente
- **Considere o kerf**: Lembre-se que os cortes serão ligeiramente mais largos que os contornos

## Enquadramento com Câmera

Se sua máquina tem suporte a câmera, você pode:

1. **Capture a imagem da câmera** do posicionamento do material
2. **Sobreponha o design** na visualização da câmera
3. **Ajuste a posição** virtualmente antes de enquadrar
4. **Enquadre para confirmar** o alinhamento físico

Veja [Integração com Câmera](../machine/camera) para detalhes.

## Solução de Problemas

**Enquadramento não corresponde ao design**: Verifique a origem do trabalho e configurações do sistema de coordenadas

**Laser dispara durante enquadramento**: Desative a potência de enquadramento ou verifique as configurações da máquina

**Enquadramento muito rápido para ver**: Reduza a velocidade de enquadramento nas configurações

**Cabeça não alcança os cantos**: Verifique se o design está dentro da área de trabalho da máquina

## Notas de Segurança

- **Nunca deixe a máquina desassistida** durante o enquadramento
- **Verifique se o laser está desligado** se estiver usando enquadramento de potência zero
- **Mantenha as mãos longe** do caminho da cabeça do laser
- **Observe obstruções** que possam interferir com o movimento

## Tópicos Relacionados

- [Integração com Câmera](../machine/camera)
- [Guia de Início Rápido](../getting-started/quick-start)
