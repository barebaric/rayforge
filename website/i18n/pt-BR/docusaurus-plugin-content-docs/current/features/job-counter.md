---
description:
  "Um contador de trabalhos em modelos de texto: um número do projeto que avança após cada trabalho, com
  números consecutivos para as cópias de uma matriz."
---

# Contador de trabalhos

As caixas de texto resolvem modelos `{...}` (veja
[Parâmetros e expressões](sketcher/expressions.md#template-expressions-in-text-boxes)). Além dos
parâmetros do esboço e das funções integradas, o projeto tem um contador de trabalhos que muda de um
trabalho para outro, para você gravar placas, etiquetas e rótulos numerados sem editar o texto a
cada vez.

## O contador de trabalhos

Use `{counter}` em uma caixa de texto. É um número inteiro, então os especificadores de formato
controlam sua aparência:

- `{counter}` — `7`
- `{counter:04d}` — `0007`, quatro dígitos com zeros à esquerda
- `No. {counter:03d}` — `No. 007`

Abra **Ferramentas → Contador de trabalhos…** para configurar o contador:

- **Valor atual** — o número com que o próximo trabalho começa.
- **Valor inicial** — para onde **Redefinir** volta.
- **Avançar em** — o incremento por cópia e por trabalho.
- **Recomeçar no valor final** / **Valor final** — depois do valor final, o contador continua no
  valor inicial.
- **Avançar após cada trabalho** — quando um trabalho enviado à máquina termina, o contador avança.
  Enquadramento, testes a seco com o ponteiro e trabalhos cancelados não o alteram.
- **Avançar após exportar G-code** — avançar também ao exportar G-code, para trabalhos executados a
  partir de um cartão de memória.

O grupo **Próximo trabalho** mostra uma prévia dos números que o próximo trabalho usará.
**Anterior** e **Próximo** movem o contador em um trabalho, **Redefinir** volta ao valor inicial.
Cada alteração, incluindo o avanço automático, faz parte do histórico de desfazer e é salva no
projeto.

## Numerar cópias

Cada peça tem um **Deslocamento de variável**. Uma cópia com deslocamento _n_ mostra o número _n_
cópias depois do atual. Uma [matriz em grade ou circular](arrays.md) dá às suas cópias deslocamentos
crescentes automaticamente, então uma matriz de 20 placas grava 20 números consecutivos, e o próximo
trabalho continua com o 21º.

Matrizes criadas dentro de um esboço são diferentes: suas cópias pertencem a uma única peça e todas
mostram o mesmo número. Crie a matriz na tela para numerar as cópias.

Para cópias que você mesmo posicionou, selecione todas e escolha **Organizar → Matriz → Numerar
cópias**: os deslocamentos são atribuídos em ordem de leitura, linhas de cima para baixo e cada
linha da esquerda para a direita.

O contador avança pelo número de cópias do trabalho: o maior deslocamento de variável mais um.
