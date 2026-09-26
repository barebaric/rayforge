# Configurações

![Configurações Gerais](/screenshots/app-settings-general.webp)

Personalize o Rayforge para corresponder ao seu fluxo de trabalho e preferências. Abra as
configurações via **Editar → Configurações** ou pressione <kbd>ctrl+vírgula</kbd>.

## Geral

A página Geral contém configurações gerais do aplicativo.

### Aparência

Escolha entre o tema **Sistema**, **Claro** ou **Escuro** para corresponder ao seu ambiente de
desktop ou preferência pessoal. Você também pode configurar as **Cores da operação** para usar a cor
do laser ou a cor da camada como distinção visual na tela.

### Unidades

Configure as unidades de exibição usadas em todo o aplicativo. Você pode definir unidades separadas
para **comprimento** (milímetros, polegadas, etc.), **velocidade** (mm/min, mm/seg, polegadas/min,
etc.) e **aceleração** (mm/s², etc.).

### Comportamento

Por padrão, as operações são recalculadas automaticamente após cada alteração. Se você trabalha em
uma máquina mais lenta ou com documentos muito complexos, pode desabilitar **Atualizar operações
automaticamente** e acionar o recálculo manualmente pelo botão na barra de ferramentas.

O Rayforge pode **Verificar atualizações** automaticamente na inicialização. Quando habilitado, você
será notificado quando uma nova versão estiver disponível.

Você também pode configurar o **Comportamento de inicialização** — iniciar com um espaço de trabalho
vazio, reabrir o último projeto ou sempre abrir um arquivo de projeto específico. Note que arquivos
especificados na linha de comando sempre substituirão essas configurações.

### Privacidade

O Rayforge pode enviar dados de uso anônimos para ajudar a melhorar o aplicativo. Nenhuma informação
pessoal é coletada. Você pode ativar ou desativar **Relatar uso anônimo** a qualquer momento. Veja a
página de [rastreamento de uso](https://rayforge.org/docs/general-info/usage-tracking) para saber
mais sobre quais dados são coletados e como são usados.

## Gestos do mouse

A página de gestos do mouse permite reatribuir os gestos de navegação da tela 2D, da tela 3D e do
editor de esboços. Cada entrada mostra a combinação de botões do mouse atribuída atualmente. Clique
em uma entrada para capturar uma nova atribuição: pressione o botão do mouse desejado (com ou sem
teclas modificadoras pressionadas) ou use a roda do mouse, e a atribuição é aplicada imediatamente.

- **Deslocar a visão** — mantenha o botão do mouse atribuído pressionado e mova para deslocar.
- **Zoom da visão** — use a roda do mouse para dar zoom.
- **Órbita / rotação (tela 3D)** — arraste para orbitar ao redor da cena ou girar ao redor do eixo
  Z.
- **Abrir o menu de contexto** — o menu de contexto da tela 2D ou o menu de ferramentas do editor de
  esboços.
- **Redefinir a visão** — ajusta a visão novamente. Sem atribuição por padrão.

Uma atribuição pode ser removida com **Remover atribuição** ou restaurada com **Restaurar padrão**.
A atribuição de um gesto que já é usado por outra ação na mesma tela é rejeitada, de modo que uma
combinação de botões do mouse nunca acione duas ações ao mesmo tempo. Addons podem contribuir com
configurações de gestos adicionais, que aparecem como seções extras nesta página.

## Outras configurações

O diálogo de configurações também inclui páginas para gerenciar outras partes do aplicativo. Cada
uma possui sua própria documentação:

- [Máquinas](../application-settings/machines.md) — adicionar, remover e configurar suas cortadoras
  a laser
- [Materiais](../application-settings/materials.md) — gerenciar suas bibliotecas de materiais
- [Receitas](../application-settings/recipes.md) — gerenciar receitas de operações salvas
- [Regras de Cor](../application-settings/color-rules.md) — mapear cores SVG para tipos de etapa
- [Provedores de IA](../application-settings/ai-provider.md) — configurar provedores de IA para uso
  pelos addons
- [Addons](../application-settings/addons.md) — instalar, atualizar e remover addons de extensão
