# Enquadramento do seu trabalho

Aprenda a usar o recurso de enquadramento para visualizar os limites do seu
trabalho e garantir o alinhamento correto antes de cortar.

## Visão geral

O enquadramento permite visualizar os limites exatos do seu trabalho a laser
traçando um contorno com o laser em baixa potência ou com o laser desligado.
Isso ajuda a verificar o posicionamento e evitar erros custosos.

## Quando usar o enquadramento

- **Primeiras configurações**: Verificar o posicionamento do material
- **Posicionamento preciso**: Garantir que o design caiba nos limites do
  material
- **Múltiplos trabalhos**: Confirmar o alinhamento antes de cada execução
- **Materiais caros**: Verificar antes de realizar os cortes

## Como enquadrar

### Método 1: Apenas contorno

Traçar o limite do trabalho sem ligar o laser:

1. **Carregue seu design** no Rayforge
2. **Posicione o material** na mesa do laser
3. **Clique no botão Enquadrar** na barra de ferramentas
4. **Observe a cabeça do laser** traçar o retângulo delimitador
5. **Verifique o posicionamento** e ajuste o material se necessário

### Método 2: Visualização em baixa potência

Algumas máquinas suportam enquadramento em baixa potência com um feixe visível:

1. **Ative o modo de baixa potência** nas configurações da máquina
2. **Defina a potência de enquadramento** (tipicamente 1-5 %)
3. **Execute a operação de enquadramento**
4. **Observe o contorno** traçado na superfície do material

:::warning Verifique sua máquina
Nem todos os lasers suportam enquadramento em baixa potência com segurança.
Consulte a documentação da sua máquina antes de usar este recurso.
:::

## Configurações de enquadramento

Configure o comportamento do enquadramento nas configurações da cabeça do
laser da sua máquina:

- **Velocidade de enquadramento**: Quão rápido a cabeça do laser se move
  durante o enquadramento. É definida por cabeça do laser, então se sua
  máquina tem múltiplos lasers você pode usar velocidades diferentes para cada
  um.
- **Potência de enquadramento**: Potência do laser durante o enquadramento
  (0 para desligado, % baixa para traço visível)
- **Tempo de pausa nos cantos**: Uma breve pausa em cada canto do contorno do
  enquadramento. Isso lhe dá um momento para ver exatamente onde cada canto
  fica — especialmente útil em velocidades de enquadramento mais altas.
- **Número de repetições**: Quantas vezes o contorno é traçado. Definir um
  valor maior que um pode tornar o caminho mais fácil de seguir visualmente.

## Usando os resultados do enquadramento

Após o enquadramento, você pode:

- **Ajustar a posição do material** se necessário
- **Reenquadrar** para verificar a nova posição
- **Prosseguir com o trabalho** uma vez satisfeito com o posicionamento

## Dicas para um enquadramento eficaz

- **Marque os cantos**: Coloque pequenos pedaços de fita nos cantos como
  referência
- **Verifique o espaço**: Garanta espaço adequado ao redor do seu design
- **Confirme a orientação**: Verifique se o material está orientado
  corretamente
- **Considere a folga de corte**: Lembre-se que os cortes serão ligeiramente
  mais largos que os contornos

## Enquadramento com câmera

Se sua máquina possui suporte para câmera, você pode:

1. **Capturar imagem da câmera** do posicionamento do material
2. **Sobrepor o design** na visualização da câmera
3. **Ajustar a posição** virtualmente antes de enquadrar
4. **Enquadrar para confirmar** o alinhamento físico

Consulte [Integração com câmera](../machine/camera) para detalhes.

## Solução de problemas

**O quadro não corresponde ao design**: Verifique a origem do trabalho e as
configurações do sistema de coordenadas

**O laser dispara durante o enquadramento**: Desative a potência de
enquadramento ou verifique as configurações da máquina

**O quadro está rápido demais para ver**: Reduza a velocidade de
enquadramento nas configurações

**A cabeça não alcança os cantos**: Verifique se o design está dentro da área
de trabalho da máquina

## Notas de segurança

- **Nunca deixe a máquina sem supervisão** durante o enquadramento
- **Verifique se o laser está desligado** ao usar enquadramento sem potência
- **Mantenha as mãos afastadas** do caminho da cabeça do laser
- **Fique atento a obstruções** que possam interferir no movimento

## Tópicos relacionados

- [Posicionamento de peça](workpiece-positioning) - Guia completo de
  posicionamento
- [Integração com câmera](../machine/camera)
- [Guia de início rápido](../getting-started/quick-start)
