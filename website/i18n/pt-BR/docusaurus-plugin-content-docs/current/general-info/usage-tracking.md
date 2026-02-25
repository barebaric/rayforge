# Rastreamento de uso

O Rayforge inclui rastreamento de uso anônimo opcional para nos ajudar a entender como o aplicativo é usado e priorizar o desenvolvimento futuro. Esta página explica o que rastreamos, como funciona e sua privacidade.

## Totalmente opcional

O rastreamento de uso é **completamente opcional**. Ao iniciar o Rayforge pela primeira vez, você será perguntado se deseja participar:

- **Sim**: Dados de uso anônimos serão enviados ao nosso servidor de análise
- **Não**: Nenhum dado é nunca coletado ou transmitido

Você pode alterar essa escolha a qualquer momento nas configurações gerais.

## O que rastreamos

Quando ativado, coletamos apenas dados anônimos de visualizações de página - semelhante à análise de sites. Aqui está o que podemos ver:

| Dados                       | Exemplo                   |
| --------------------------- | ------------------------- |
| Resolução de tela           | 1920x1080                 |
| Configuração de idioma      | pt-BR                     |
| Páginas/diálogos visualizados | /machine-settings/general |
| Tempo na página             | 6m 3s                     |

## O que vemos

Aqui está um exemplo de como o painel de análise se parece:

| Caminho                      | Visitantes | Visitas | Visualizações | Taxa de rejeição | Duração da visita |
| ---------------------------- | ---------- | ------- | ------------- | ---------------- | ----------------- |
| /                            | 1          | 1       | 5             | 0%               | 27m 35s           |
| /machine-settings/general    | 1          | 1       | 5             | 0%               | 27m 27s           |
| /view/3d                     | 1          | 1       | 2             | 0%               | 25m 14s           |
| /camera-alignment-dialog     | 1          | 1       | 2             | 0%               | 6m 3s             |
| /machine-settings/camera     | 1          | 1       | 2             | 0%               | 6m 16s            |
| /settings/general            | 1          | 1       | 2             | 0%               | 16m 36s           |
| /step-settings/rasterizer    | 1          | 1       | 2             | 0%               | 11s               |

## O que NÃO rastreamos

Estamos comprometidos com sua privacidade:

- **Sem informações pessoais** – Sem nomes, e-mails ou contas
- **Sem conteúdo de arquivos** – Seus designs e projetos permanecem privados
- **Sem identificadores de máquina** – Sem números de série ou IDs únicos
- **Sem endereços IP armazenados** – Usamos Umami Analytics que não armazena IPs
- **Sem rastreamento entre sites** – Os dados são isolados apenas ao Rayforge

## Por que rastreamos

Os dados de uso nos ajudam a:

- **Identificar recursos populares** – Saber o que está funcionando bem
- **Encontrar pontos problemáticos** – Ver onde os usuários gastam tempo ou ficam presos
- **Priorizar o desenvolvimento** – Focar em recursos que as pessoas realmente usam
- **Entender a diversidade** – Saber quais idiomas e tamanhos de tela suportar

## Como funciona

O Rayforge usa [Umami](https://umami.is/), uma plataforma de análise de código aberto focada em privacidade. O rastreamento:

- Envia pequenas requisições HTTP em segundo plano
- Não afeta o desempenho do aplicativo
- Funciona offline (requisições com falha são silenciosamente ignoradas)
- Usa um User-Agent genérico para evitar fingerprinting

## Desativar o rastreamento

Você pode desativar o rastreamento a qualquer momento:

1. Abra **Configurações** → **Geral**
2. Desative **Enviar estatísticas de uso anônimas**

Quando desativado, absolutamente nenhum dado é enviado.

## Páginas relacionadas

- **[Configurações do aplicativo](../ui/settings)** – Configurar preferências de rastreamento
