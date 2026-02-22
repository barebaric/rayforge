# Solução de Problemas e Relatando Problemas

Se você está enfrentando problemas com o Rayforge, especialmente com conectar ou controlar sua máquina, estamos aqui para ajudar. A melhor maneira de obter suporte é fornecendo um relatório de depuração detalhado. O Rayforge tem uma ferramenta integrada que torna isso fácil.

## Como Criar um Relatório de Depuração

Siga estes passos simples para gerar e compartilhar um relatório:

#### 1. Salvar o Relatório

Vá para **Ajuda → Salvar Log de Depuração** na barra de menu. Isso empacotará toda a informação de diagnóstico necessária em um único arquivo `.zip`. Salve este arquivo em um local memorável, como sua Área de Trabalho.

#### 2. Criar uma Issue no GitHub

Vá para nossa [página de GitHub Issues](https://github.com/barebaric/rayforge/issues/new/choose) e crie uma nova issue. Por favor forneça um título claro e uma descrição detalhada do problema:

- **O que você fez?** (ex: "Tentei conectar ao meu laser após iniciar o app.")
- **O que você esperava que acontecesse?** (ex: "Esperava que conectasse com sucesso.")
- **O que realmente aconteceu?** (ex: "Permaneceu desconectado e o log mostrou erros de timeout.")

#### 3. Anexar o Relatório

**Arraste e solte o arquivo `.zip`** que você salvou na caixa de descrição da issue do GitHub. Isso fará upload e anexará ao seu relatório.

## O Que Está no Relatório de Depuração?

O arquivo `.zip` gerado contém informação técnica que nos ajuda diagnosticar o problema rapidamente. Ele inclui:

- **Configurações da Máquina e Aplicativo:** Suas configurações de máquina salvas e preferências do aplicativo, o que nos ajuda a reproduzir sua configuração.
- **Logs de Comunicação:** Um registro detalhado dos dados enviados entre o Rayforge e seu laser.
- **Informação do Sistema:** Seu sistema operacional e as versões do Rayforge e bibliotecas principais instaladas.
- **Estado da Aplicação:** Outra informação interna que pode ajudar a identificar a fonte de um erro.

> **Nota de Privacidade:** O relatório **não** inclui nenhum dos seus arquivos de design (SVGs, DXFs, etc.) ou dados pessoais do sistema operacional. Ele contém apenas informação diretamente relacionada ao aplicativo Rayforge e sua conexão com seu laser.

Obrigado por nos ajudar a melhorar o Rayforge
