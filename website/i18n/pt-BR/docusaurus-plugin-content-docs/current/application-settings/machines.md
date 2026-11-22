---
description:
  "Gerencie máquinas no Rayforge — adicione, configure, exporte, importe e alterne entre diferentes
  cortadores e gravadores a laser para seus projetos."
---

# Máquinas

![Configurações de Máquinas](/screenshots/app-settings-machines.webp)

A página **Máquinas** mostra uma lista de todas as máquinas configuradas. Cada entrada mostra o nome
da máquina e possui botões para editá-la ou excluí-la. A máquina ativa no momento é marcada com um
ícone de verificação.

Abra a página na janela principal com **Máquina → Manage Machines…**, ou selecione **Máquinas** nas
Configurações da Aplicação (<kbd>ctrl+comma</kbd>).

## Adicionar uma Máquina

Na página Máquinas, clique no botão **Add Machine** na parte inferior da lista e então:

1. Selecione um perfil de dispositivo da lista para usar como modelo — cada perfil pré-configura as
   definições da máquina e o dialeto de G-code

![Adicionar Máquina](/screenshots/app-settings-machines-add.webp)

2. O [diálogo de configurações da máquina](../machine/general.md) abre onde você pode ajustar a
   configuração

Alternativamente:

- Clique em **Device Not Listed** para iniciar o
  [Assistente de Configuração](../getting-started/first-time-setup.md), que orienta você na
  configuração de uma máquina passo a passo
- Clique em **Import from File…** para adicionar uma máquina de um perfil exportado anteriormente ou
  de um perfil de dispositivo LightBurn (.lbdev). Perfis LightBurn incluem calibração de câmera e
  configurações de laser que são aplicadas à nova máquina.

## Editar uma Máquina

Clique no ícone de edição ao lado de uma máquina para abrir o
[diálogo de configurações da máquina](../machine/general.md).

## Notas da máquina

A categoria **Notes**, logo após **General**, reúne as orientações do perfil do dispositivo e suas
próprias notas para essa máquina. As notas do dispositivo são somente leitura; **My Notes** podem
ser editadas com Markdown. A sintaxe compatível inclui títulos, parágrafos, negrito, itálico,
listas, links, código em linha, blocos de código e seções expansíveis com `:::details`.

Quando um perfil fornece instruções de configuração, elas também aparecem na página final **Review &
Name** do assistente e permanecem disponíveis em **Machine Settings → Notes** após a criação da
máquina. As notas pessoais são salvas por máquina e não são substituídas ao revisar atualizações do
perfil.

## Alternar a Máquina Ativa

O submenu **Máquina → Select Machine** lista todas as máquinas configuradas; a máquina ativa tem uma
marca de verificação. Selecione outra entrada para alternar para ela. Você também pode usar o menu suspenso
de máquinas no cabeçalho da janela principal. A seleção é lembrada entre sessões.

## Excluir uma Máquina

1. Clique no ícone de exclusão ao lado da máquina
2. Confirme a exclusão

<!-- prettier-ignore-start -->
:::warning
Excluir uma máquina não pode ser desfeito. Exporte o perfil primeiro se desejar preservar
a configuração.
:::
<!-- prettier-ignore-end -->
