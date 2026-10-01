---
description:
  "A etapa de comando injeta código de máquina personalizado, uma linha por comando, em uma posição
  exata do fluxo de trabalho de uma camada. Use para pré-posicionamento, controle de acessórios e
  comandos específicos da máquina."
---

# Comando

A etapa de comando injeta código de máquina personalizado no trabalho, exatamente na posição que a
etapa ocupa no fluxo de trabalho da camada. Use-a para enviar comandos que as operações de geometria
não cobrem: posicionar o cabeçote, ligar ou desligar a assistência de ar ou outros equipamentos em
torno de operações específicas, ou enviar códigos específicos da máquina.

![Configurações da etapa de comando](/screenshots/step-settings-command-general.webp)

## Visão geral

A etapa de comando:

- Armazena um bloco de código de máquina com várias linhas, um comando por linha
- Emite cada linha literalmente na posição da etapa quando o trabalho é codificado
- É executada uma vez por camada, em sua posição no fluxo de trabalho — ela não age sobre as peças
- Funciona sem nenhuma peça na camada, de modo que uma camada contendo apenas uma etapa de comando
  ainda pode gerar um trabalho
- Suporta as mesmas variáveis de caminho que as macros (veja abaixo)

O texto é armazenado sem expansão no projeto, então viaja com o arquivo `.ryp` e documenta
exatamente o que é enviado e para onde.

## Quando usar a etapa de comando

Use a etapa de comando para:

- Pré-posicionar o cabeçote (ex: elevar Z antes do início de um corte)
- Ligar ou desligar a assistência de ar, o refrigerante ou outros equipamentos entre operações
- Enviar códigos específicos da controladora em torno de um trabalho
- Pausar brevemente com um dwell entre duas operações

**Não use a etapa de comando para:**

- Repetir código G nos limites de camada ou de peça — as
  [Macros & Hooks](../../machine/hooks-macros.md) são acionadas automaticamente e também funcionam
  em máquinas sem código G
- Executar programas no computador (isso não é suportado nesta fase)

## Adicionar uma etapa de comando

1. Abra o fluxo de trabalho da camada no painel direito.
2. Clique no botão **Adicionar etapa** e escolha **Comando**.
3. Digite o código de máquina na caixa de texto, um comando por linha. Linhas vazias são ignoradas.

A etapa pode ser colocada antes, entre ou depois de outras etapas, e pode ser usada várias vezes em
uma camada. Sua posição no fluxo de trabalho é a posição que suas linhas ocupam no código de máquina
gerado.

## Variáveis de caminho

Como as macros, as linhas podem conter variáveis de caminho que são resolvidas quando o trabalho é
codificado. Variáveis desconhecidas permanecem intactas, e as variáveis `layer.*` são resolvidas
mesmo quando a etapa é executada no meio de uma camada.

| Variável           | Exemplo    | Descrição                                                  |
| ------------------ | ---------- | ---------------------------------------------------------- |
| `{machine.name}`   | `My Laser` | Nome da máquina ativa                                      |
| `{layer.name}`     | `Layer 1`  | Nome da camada que está sendo processada                   |
| `{job.extents[0]}` | `210.0`    | Extensão do trabalho no eixo X (mm)                        |
| `{wcs_offset[0]}`  | `5.0`      | Deslocamento X do sistema de coordenadas de trabalho ativo |

Por exemplo, um comentário como `; cortando {layer.name} na {machine.name}` é codificado com os
nomes preenchidos.

## Suporte a máquinas

A etapa de comando está disponível em todas as máquinas, mas as linhas são emitidas apenas para os
drivers que consomem código de máquina. Quando o driver da máquina ativa não consome (ex: Ruida), a
etapa mostra um aviso, e suas linhas não fazem parte da saída gerada.
