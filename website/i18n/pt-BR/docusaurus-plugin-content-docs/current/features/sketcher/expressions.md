---
description:
  "Parâmetros de esboço, expressões de restrição e modelos de caixa de texto no esboçador do
  Rayforge: guiando a geometria e os rótulos com valores nomeados e fórmulas."
---

# Expressões e Parâmetros

Um esboço se torna verdadeiramente paramétrico quando suas dimensões são guiadas por valores
nomeados em vez de números fixos. O esboçador suporta isso em dois lugares: restrições dimensionais
aceitam **expressões**, e caixas de texto aceitam **expressões de modelo**. Ambas são avaliadas pelo
solver, de modo que o esboço se atualiza automaticamente sempre que um valor muda.

## Parâmetros do esboço

Cada esboço carrega sua própria lista de parâmetros, exibida no painel **Parâmetros do Esboço** à
esquerda do editor de esboços. **Adicionar Parâmetro** cria um, com escolha entre inteiro, número de
ponto flutuante, controle deslizante ou uma única linha de texto. Cada parâmetro tem um nome — a
coluna `key` — e esse nome é o que as expressões referenciam.

Uma configuração típica para uma caixa com espessura de parede variável consiste em dois parâmetros,
`width` e `thickness`. Nada restringe a geometria ainda; os parâmetros são apenas nomes para números
até que uma expressão os use.

## Expressões em restrições

Dê um duplo clique em uma restrição dimensional (veja [Restrições](constraints.md)) e informe uma
expressão em vez de um número simples:

```
width / 2
```

O valor da restrição se torna o resultado dessa expressão, reavaliado toda vez que o esboço é
resolvido. Altere o parâmetro `width` e a geometria restringida acompanha — uma única edição agora
atualiza todas as dimensões que a referenciam. Restrições guiadas por uma expressão desenham seu
marcador em laranja, e o rótulo mostra o valor calculado.

Expressões podem combinar parâmetros com aritmética e as funções matemáticas padrão do Python:

```
width - 2 * thickness
sqrt(area) / 2
2 * pi * radius
```

Funções como `sqrt`, `sin`, `cos` e `tan`, e constantes como `pi`, vêm do módulo `math` do Python —
esse módulo, mais os parâmetros, é exatamente o que uma expressão de restrição pode referenciar.
Parâmetros do tipo string também podem ser referenciados, o que é útil principalmente em caixas de
texto.

## Expressões de modelo em caixas de texto {#template-expressions-in-text-boxes}

As caixas de texto resolvem expressões entre chaves no momento da resolução, de modo que rótulos e
textos gravados exibem valores ao vivo:

```
W = {width}, H = {height}
```

Qualquer parâmetro pode ser substituído pelo nome, e o resultado pode ser formatado com um
especificador de formato Python após dois-pontos:

- `{width}` — o valor atual do parâmetro `width`
- `{name}` — o valor de um parâmetro do tipo string
- `{width:.1f}` — uma casa decimal
- `{timestamp():.0f}` — sem casas decimais no resultado de uma função

Matemática também funciona aqui, seja como uma expressão como `{width * 2}` ou por meio de uma
função como `{sqrt(area):.2f}`. Em comparação com as expressões de restrição, os modelos de texto
têm uma caixa de ferramentas mais rica: junto com o módulo matemático, eles expõem as funções
integradas abaixo, e funções personalizadas podem ser registradas para eles (veja
[abaixo](#custom-template-functions)).

### Funções integradas de modelo

| Função          | Tipo de retorno | Descrição                                            |
| --------------- | --------------- | ---------------------------------------------------- |
| `{today()}`     | `date`          | Data UTC atual (ex.: `2026-08-26`)                   |
| `{date()}`      | `date`          | Alias de `today()`                                   |
| `{now()}`       | `datetime`      | Data e hora UTC atuais                               |
| `{time()}`      | `time`          | Hora UTC atual (ex.: `15:30:00.123456+00:00`)        |
| `{timestamp()}` | `float`         | Marca temporal Unix (segundos desde a época)         |
| `{uuid4()}`     | `str`           | String hexadecimal de 8 caracteres (ex.: `a1b2c3d4`) |
| `{uuid8()}`     | `str`           | Alias de `uuid4()`                                   |
| `{uuid()}`      | `str`           | String UUID v4 completa (36 caracteres)              |

Usos típicos incluem números de série únicos a cada resolução (`Peça #{uuid4()}`), rótulos de
dimensões em tempo real (`L={width:.1f} A={height:.1f}`), datar cada peça (`Data: {today()}`),
contadores de produção (`{name} - {count:.0f}un`) ou marcas temporais Unix para registro de produção
(`{timestamp():.0f}`).

## Funções de modelo personalizadas {#custom-template-functions}

Você pode registrar suas próprias funções para usar dentro de modelos de texto. Isso é útil para
obter números de série de um banco de dados, ler dados externos ou gerar rótulos personalizados.

### Escrever o script de registro

Crie um arquivo Python (ex. `~/.config/rayforge/minhas_funcoes.py`):

```python
"""Registrar funções personalizadas para modelos de texto."""
import sqlite3

from sketcher.core.template_functions import (
    register_template_function,
)

CAMINHO_DB = "/home/voce/producao.db"


def proximo_serial() -> str:
    """Obter próximo número de série do banco."""
    conn = sqlite3.connect(CAMINHO_DB)
    try:
        cur = conn.execute(
            "UPDATE contadores SET valor = valor + 1 "
            "WHERE nome = 'serial' RETURNING valor"
        )
        row = cur.fetchone()
        conn.commit()
        return f"SN-{row[0]:06d}"
    finally:
        conn.close()


register_template_function("proximo_serial", proximo_serial)
```

Chame `register_template_function(nome, callable)` para cada função. A função pode fazer qualquer
coisa que o Python possa — abrir arquivos, conectar a bancos de dados, chamar APIs — e ela é chamada
em **cada renderização**, então deve ser rápida (use cache se os dados subjacentes não mudarem entre
renderizações). As funções são thread-safe se o seu callable for.

### Executar Rayforge com o script

Use a flag `--script` para carregar suas funções antes da abertura da janela:

```bash
rayforge --script ~/.config/rayforge/minhas_funcoes.py \
    meu_documento.ryp
```

Isso executa seu script no início da inicialização — antes dos addons serem carregados e antes da
janela principal ser criada — para que a função esteja disponível quando o esboço for resolvido pela
primeira vez.

### Usar a função em uma caixa de texto

No esboçador, crie uma caixa de texto com:

```
{proximo_serial()}
```

Especificações de formato também funcionam:

```
{proximo_serial():>20}
```

### Registrar funções programaticamente

Se você está escrevendo um addon ou biblioteca reutilizável, pode chamar
`register_template_function` de qualquer código Python que execute antes da resolução do esboço:

```python
from sketcher.core.template_functions import (
    register_template_function,
)

register_template_function(
    "numero_peca",
    lambda: f"P-{hash('x') % 10000:04d}"
)
```

### Funções integradas não podem ser removidas

As funções integradas (`today`, `now`, `uuid`, etc.) não podem ser desregistradas. Se você precisar
alterar seu comportamento, registre uma função com um nome diferente.
