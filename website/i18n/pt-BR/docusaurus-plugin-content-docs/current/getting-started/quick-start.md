# Guia de Início Rápido

Agora que o Rayforge está instalado e sua máquina está configurada, vamos executar seu primeiro trabalho de laser! Este guia irá orientá-lo na importação de um design, configuração de operações e envio de G-code para sua máquina.

## Passo 1: Importar um Design

O Rayforge suporta vários formatos de arquivo, incluindo SVG, DXF, PDF e imagens raster (JPEG, PNG, BMP).

1. **Clique** em **Arquivo → Abrir** ou pressione <kbd>ctrl+o</kbd>
2. Navegue até o arquivo de design e selecione-o
3. O design aparecerá na tela

![Tela com design importado](/screenshots/main-standard.png)

:::tip Não tem um design ainda?
Você pode criar formas simples usando as ferramentas da tela ou baixar arquivos SVG gratuitos de sites como [Flaticon](https://www.flaticon.com/) ou [SVG Repo](https://www.svgrepo.com/).
:::


## Passo 2: Posicionar Seu Design

Use as ferramentas da tela para posicionar e ajustar seu design:

- **Pan**: Clique com o botão do meio e arraste, ou segure <kbd>espaço</kbd> e arraste
- **Zoom**: Roda do mouse, ou <kbd>ctrl+"+"</kbd> / <kbd>ctrl+"-"</kbd>
- **Mover**: Clique e arraste seu design
- **Rotacionar**: Selecione o design e use as alças de rotação
- **Escalar**: Selecione o design e arraste as alças dos cantos

## Passo 3: Atribuir uma Operação

As operações definem como o Rayforge processará seu design. As operações comuns incluem:

- **Contorno**: Corta ao longo da borda das formas
- **Gravação Raster**: Preenche formas com linhas de vai-e-vem (para gravação)
- **Gravação em Profundidade**: Cria efeitos de profundidade 3D a partir de imagens

### Adicionando uma Operação

1. Selecione seu design na tela
2. Clique em **Operações → Adicionar Operação** ou pressione <kbd>ctrl+shift+a</kbd>
3. Escolha o tipo de operação (ex: "Contorno" para corte)
4. Configure as definições da operação:
   - **Potência**: Porcentagem de potência do laser (comece baixo e teste!)
   - **Velocidade**: Velocidade de movimento em mm/min
   - **Passagens**: Número de vezes para repetir a operação (útil para cortar materiais espessos)

![Configurações de Operação](/screenshots/step-settings-contour-general.png)

:::warning Comece com Potência Baixa
Ao trabalhar com novos materiais, sempre comece com configurações de potência mais baixas e execute cortes de teste. Aumente a potência gradualmente até atingir o resultado desejado. Use o recurso [Grade de Teste de Material](../features/operations/material-test-grid) para encontrar sistematicamente as configurações ideais.
:::


## Passo 4: Visualizar

Antes de enviar para sua máquina, visualize o caminho da ferramenta em 3D:

1. Clique em **Visualizar → Visualização 3D** ou pressione <kbd>ctrl+3</kbd>
2. A janela de visualização 3D mostra o caminho completo da ferramenta
3. Use o mouse para rotacionar e dar zoom na visualização
4. Verifique se o caminho parece correto

![Visualização 3D](/screenshots/main-3d.png)

:::tip Detecte Erros Cedo
A visualização 3D ajuda você a identificar problemas como:

- Caminhos faltando
- Ordem incorreta
- Operações aplicadas aos objetos errados
- Caminhos que excedem sua área de trabalho
:::


## Passo 5: Enviar para a Máquina

:::danger Segurança em Primeiro Lugar
- Certifique-se de que a área de trabalho está livre
- Nunca deixe a máquina sem supervisão durante a operação
- Tenha equipamento de segurança contra incêndio por perto
- Use proteção ocular apropriada
:::


### Preparando Seu Material

1. Coloque seu material na mesa do laser
2. Focalize o laser de acordo com as instruções da sua máquina
3. Se estiver usando a câmera, alinhe seu design usando a [sobreposição de câmera](../machine/camera)

### Iniciando o Trabalho

1. **Posicione o laser**: Use os controles de jog para mover o laser para a posição inicial
   - Clique em **Visualizar → Painel de Controle** ou pressione <kbd>ctrl+l</kbd>
   - Use os botões de seta ou as setas do teclado para mover o laser
   - Pressione <kbd>home</kbd> para levar a máquina à origem

2. **Enquadrar o design**: Execute a função de enquadramento para verificar o posicionamento
   - Clique em **Máquina → Enquadrar** ou pressione <kbd>ctrl+f</kbd>
   - O laser traçará a caixa delimitadora do seu design com potência baixa/zero
   - Verifique se cabe dentro do seu material

3. **Iniciar o trabalho**: Clique em **Máquina → Iniciar Trabalho** ou pressione <kbd>ctrl+r</kbd>
4. Monitore o progresso na barra de status

### Durante o Trabalho

- A seção direita da barra de status mostra o progresso atual e estimativa de tempo total de execução
- Você pode pausar o trabalho com <kbd>ctrl+p</kbd> ou clicar no botão Pausar
- Pressione <kbd>esc</kbd> ou clique em Parar para cancelar o trabalho (parada de emergência)

## Passo 6: Finalizando

Uma vez que o trabalho é concluído:

1. Espere o exaustor limpar qualquer fumaça
2. Remova cuidadosamente sua peça finalizada
3. Limpe a mesa do laser se necessário

:::success Parabéns!
Você completou seu primeiro trabalho no Rayforge! Agora você pode explorar recursos mais avançados.
:::


## Próximos Passos

Agora que você completou seu primeiro trabalho, explore estes recursos:

- **[Operações Multi-Camadas](../features/multi-layer)**: Atribua operações diferentes às camadas
- **[Abas de Fixação](../features/holding-tabs)**: Mantenha peças cortadas no lugar durante o corte
- **[Integração com Câmera](../machine/camera)**: Use uma câmera para alinhamento preciso
- **[Hooks & Macros](../machine/hooks-macros)**: Automatize tarefas repetitivas

## Dicas para o Sucesso

1. **Salve seu trabalho**: Use <kbd>ctrl+s</kbd> para salvar seu projeto frequentemente
2. **Cortes de teste**: Sempre execute um corte de teste em material de sobra primeiro
3. **Banco de dados de materiais**: Mantenha anotações de configurações de potência/velocidade bem-sucedidas para diferentes materiais
4. **Manutenção**: Mantenha a lente do laser limpa e verifique a tensão das correias regularmente
5. **Assistência de ar**: Se sua máquina tem assistência de ar, use-a para prevenir queima e melhorar a qualidade do corte

---

**Precisa de Ajuda?** Verifique a seção [Solução de Problemas](../troubleshooting/connection) ou visite a página [GitHub Issues](https://github.com/barebaric/rayforge/issues).
