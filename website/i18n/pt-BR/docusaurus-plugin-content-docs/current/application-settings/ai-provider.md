# Provedor de IA

![Configurações do Provedor de IA](/screenshots/application-ai.png)

Configure provedores de IA que os addons podem usar para adicionar
recursos inteligentes ao Rayforge.

## Como Funciona

Os addons podem usar os provedores de IA configurados sem precisar de
suas próprias chaves de API. Isso centraliza sua configuração de IA e
permite que você controle quais provedores estão disponíveis para os addons.

## Adicionar um Provedor

1. Clique em **Adicionar Provedor** para criar uma nova configuração
2. Digite um **Nome** para identificar este provedor
3. Defina a **URL Base** para o endpoint da API do seu serviço de IA
4. Digite sua **Chave de API** para autenticação
5. Especifique um **Modelo Padrão** para usar com este provedor
6. Clique em **Testar** para verificar se sua configuração funciona

## Tipos de Provedor

### Compatível com OpenAI

Este tipo de provedor funciona com qualquer serviço que use o formato
de API OpenAI. Isso inclui vários provedores em nuvem e soluções
auto-hospedadas.

A URL base padrão está definida para a API da OpenAI, mas você pode
alterá-la para qualquer serviço compatível.

## Gerenciar Provedores

- **Ativar/Desativar**: Ative ou desative um provedor sem excluí-lo
- **Definir como Padrão**: Clique no ícone de marca de seleção para
  tornar um provedor o padrão
- **Excluir**: Remova um provedor que você não precisa mais

:::warning
Suas chaves de API são armazenadas localmente em seu computador e nunca
são compartilhadas com terceiros.
:::

## Tópicos Relacionados

- [Addons](addons) - Instalar e gerenciar addons
- [Máquinas](machines) - Configuração de máquinas
- [Materiais](materials) - Bibliotecas de materiais
