---
description: "Crie cópias com os modos Grade, Rotação Pontual e Circular. Cada modo oferece visualização ao vivo e posicionamento interativo."
---

# Arrays

A funcionalidade de Array permite criar múltiplas cópias de peças de
trabalho selecionadas usando três modos de layout diferentes. Cada modo abre
um diálogo não modal, permitindo que você continue interagindo com a tela
enquanto ajusta os parâmetros — a visualização é atualizada em tempo real.

Para abrir um diálogo de array, selecione uma ou mais peças de trabalho na
tela, depois escolha o modo de array na barra de ferramentas ou no menu
contextual.

:::tip
Todos os modos de array são não modais. Você pode arrastar peças de trabalho
na tela enquanto o diálogo está aberto, e a visualização será atualizada
ao vivo para refletir as novas posições.
:::

---

## Grade

O modo Grade organiza as cópias em uma matriz retangular de linhas e
colunas, com espaçamento horizontal e vertical configurável.

![Array Grade](/screenshots/main-array-grid.png)

### Configurações

| Parâmetro | Descrição |
|-----------|-------------|
| **Linhas** | Número de linhas (1–360) |
| **Colunas** | Número de colunas (1–360) |
| **Modo de espaçamento** | Escolha entre *Espaço* (espaço entre cópias) ou *Passo* (distância de borda a borda de cada cópia) |
| **Espaçamento de colunas** | Espaçamento horizontal entre colunas |
| **Espaçamento de linhas** | Espaçamento vertical entre linhas |

---

## Rotação Pontual

O modo Rotação Pontual cria cópias girando-as no próprio centro da seleção.
Isso é útil para criar padrões circulares onde cada cópia permanece em sua
localização original mas é girada por uma fração do ângulo total.

![Array Rotação Pontual](/screenshots/main-array-point-rotation.png)

### Configurações

| Parâmetro | Descrição |
|-----------|-------------|
| **Quantidade** | Número de cópias (1–360) |
| **Ângulo total (graus)** | Extensão angular total de todas as cópias (−360° a 360°) |

:::info
Como a rotação é em torno do próprio centro da seleção, arrastar a peça
de trabalho na tela move todas as cópias juntas enquanto o diálogo permanece
aberto.
:::

---

## Circular

O modo Circular posiciona cópias ao longo de um arco circular ao redor de
um ponto central. Um marcador em cruz na tela mostra o centro, e você pode
arrastá-lo para uma nova posição enquanto o diálogo está aberto.

![Array Circular](/screenshots/main-array-circular.png)

### Configurações

| Parâmetro | Descrição |
|-----------|-------------|
| **Quantidade** | Número de cópias (1–360) |
| **Ângulo total (graus)** | Extensão angular do arco (−360° a 360°) |
| **Centro X** | Coordenada X do centro do círculo |
| **Centro Y** | Coordenada Y do centro do círculo |
| **Raio** | Raio da trajetória circular |
| **Girar cópias** | Quando habilitado, cada cópia é girada para seguir a tangente do arco |

:::tip Arrastar o centro
A cruz na tela representa o centro do círculo. Arraste-a para reposicionar
o array interativamente — os campos Centro X e Centro Y no diálogo serão
atualizados automaticamente.
:::

:::tip Arrastar peças de trabalho
Você também pode arrastar a peça de trabalho original na tela. O raio será
atualizado automaticamente para manter as cópias em sua distância atual
do centro.
:::
