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

![Configurações de gestos do mouse](/screenshots/app-settings-gestures.webp)

A página de gestos do mouse permite reatribuir os gestos de navegação da tela 2D e da tela 3D; o
editor de esboços usa os mesmos gestos da tela 2D. Cada linha oferece um menu suspenso com as
combinações de botões do mouse disponíveis:

- **Tela 2D** — _Deslocar a visão_ (por padrão, arrastar com o botão do meio; no botão esquerdo, o
  arraste é reservado para a seleção e por isso não é oferecido).
- **Tela 3D** — _Orbitar a câmera_ (arrastar com o botão do meio), _Deslocar a câmera_ (Shift +
  botão do meio) e _Girar ao redor do eixo Z_ (arrastar com o botão esquerdo).

O zoom (roda do mouse), o menu de contexto (botão direito do mouse) e a redefinição da visão (a
tecla `1` na tela 2D) são fixos e não podem ser alterados. Um clique simples em um botão atribuído
ainda executa sua ação fixa: quando o deslocamento é atribuído ao botão direito do mouse, arrastar
com o botão direito desloca a visão e um clique direito simples ainda abre o menu de contexto.
Selecionar **Sem atribuição** desativa um gesto, e combinações já usadas por outra ação da mesma
tela não são oferecidas. Addons podem contribuir com configurações de gestos adicionais, que
aparecem como seções extras nesta página.

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
