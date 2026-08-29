---
description: "Modelos de caixas de texto e funções de modelo personalizadas no esboçador paramétrico 2D do Rayforge."
---

# Modelos de Texto

As caixas de texto suportam expressões de modelo entre chaves. Elas são
resolvidas no momento da resolução usando os valores atuais dos parâmetros,
então o texto é atualizado automaticamente quando você altera uma dimensão ou
variável de entrada.

## Substituição de Variáveis

Referencie qualquer parâmetro do esboço ou variável de entrada pelo nome:

- `{width}` — o valor atual do parâmetro "width"
- `{name}` — o valor de um parâmetro de entrada do tipo string
- `{count:.0f}` — formatado com um especificador de formato Python (sem decimais)

## Expressões Matemáticas

Você pode usar funções matemáticas nos modelos:

- `{sqrt(area):.2f}` — raiz quadrada de "area", formatada com 2 casas decimais
- `{width * 2}` — expressões aritméticas

As funções matemáticas padrão (`sqrt`, `sin`, `cos`, `tan`, `pi`, etc.) estão
disponíveis.

## Funções Integradas

| Função          | Tipo retorno | Descrição                                            |
| --------------- | ------------ | ---------------------------------------------------- |
| `{today()}`     | `date`       | Data UTC atual (ex.: `2026-08-26`)                   |
| `{date()}`      | `date`       | Alias de `today()`                                   |
| `{now()}`       | `datetime`   | Data e hora UTC atuais                               |
| `{time()}`      | `time`       | Hora UTC atual (ex.: `15:30:00.123456+00:00`)        |
| `{timestamp()}` | `float`      | Marca temporal Unix (segundos desde época)           |
| `{uuid4()}`     | `str`        | String hexadecimal de 8 caracteres (ex.: `a1b2c3d4`) |
| `{uuid8()}`     | `str`        | Alias de `uuid4()`                                   |
| `{uuid()}`      | `str`        | String UUID v4 completa (36 caracteres)              |

## Especificações de Formato

Especificações de formato Python funcionam com qualquer
resultado de expressão:

- `{width:.1f}` — uma casa decimal
- `{timestamp():.0f}` — sem casas decimais na marca temporal
- `{today()}` — representação de string padrão

## Exemplos de Uso

- `Peça #{uuid4()}` — número de série único a cada resolução
- `L={width:.1f} A={height:.1f}` — rótulos de dimensões em tempo real
- `Data: {today()}` — datar cada peça
- `{name} - {count:.0f}un` — combinar parâmetros de string e numéricos
- `{timestamp():.0f}` — marca temporal Unix para registro de produção

## Funções de Modelo Personalizadas

Você pode registrar suas próprias funções para usar dentro
de modelos de texto. Isso é útil para obter números de série
de um banco de dados, ler dados externos ou gerar rótulos
personalizados.

### Escrever o script de registro

Crie um arquivo Python (ex.
`~/.config/rayforge/minhas_funcoes.py`):

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

Pontos importantes:

- Chame `register_template_function(nome, callable)` para
  cada função.
- Sua função pode fazer qualquer coisa que o Python possa:
  abrir arquivos, conectar a bancos de dados, chamar APIs,
  etc.
- A função é chamada em **cada renderização**, então deve
  ser rápida.
- As funções são thread-safe se seu callable for.

### Executar Rayforge com o script

Use a flag `--script` para carregar suas funções antes da
abertura da janela:

```bash
rayforge --script ~/.config/rayforge/minhas_funcoes.py \
    meu_documento.ryp
```

Isso executa seu script no início da inicialização — antes
dos addons serem carregados e antes da janela principal
ser criada — para que a função esteja disponível quando o
esboço for resolvido pela primeira vez.

### Usar a função em uma caixa de texto

Crie uma caixa de texto com:

```
{proximo_serial()}
```

Especificações de formato também funcionam:

```
{proximo_serial():>20}
```

### Registrar funções programaticamente

Se você está escrevendo um addon ou biblioteca reutilizável,
pode chamar `register_template_function` de qualquer código
Python que execute antes da resolução do esboço:

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

As funções integradas (`today`, `now`, `uuid`, etc.) não
podem ser desregistradas. Se você precisar alterar seu
comportamento, registre uma função com um nome diferente.
