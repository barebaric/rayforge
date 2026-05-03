---
description: "Gerencie máquinas no Rayforge — adicione, configure, exporte, importe e alterne entre diferentes cortadores e gravadores a laser para seus projetos."
---

# Máquinas

![Configurações de Máquinas](/screenshots/application-machines.png)

A página Máquinas nas Configurações da Aplicação mostra uma lista de todas
as máquinas configuradas. Cada entrada mostra o nome da máquina e possui
botões para editá-la ou excluí-la. A máquina ativa no momento é marcada com
um ícone de verificação.

## Adicionar uma Máquina

1. Clique no botão **Add Machine** na parte inferior da lista
2. Selecione um perfil de dispositivo da lista para usar como modelo — cada
   perfil pré-configura as definições da máquina e o dialeto de G-code

![Adicionar Máquina](/screenshots/add-machine-dialog.png)

3. O [diálogo de configurações da máquina](../machine/general) abre onde
   você pode ajustar a configuração

Alternativamente, clique em **Import from File...** no seletor de perfil
para adicionar uma máquina de um perfil exportado anteriormente.

## Editar uma Máquina

Clique no ícone de edição ao lado de uma máquina para abrir o
[diálogo de configurações da máquina](../machine/general).

## Alternar a Máquina Ativa

Use o menu suspenso de máquinas no cabeçalho da janela principal para
alternar entre as máquinas configuradas. A seleção é lembrada entre sessões.

## Excluir uma Máquina

1. Clique no ícone de exclusão ao lado da máquina
2. Confirme a exclusão

:::warning
Excluir uma máquina não pode ser desfeito. Exporte o perfil primeiro se
desejar preservar a configuração.
:::
