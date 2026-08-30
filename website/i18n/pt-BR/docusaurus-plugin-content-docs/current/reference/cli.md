---
description: "Referência da linha de comando do Rayforge."
---

# Linha de Comando

Referência completa das opções de linha de comando.

```
rayforge [opções] [arquivos...]
```

---

## Argumentos posicionais

| Argumento  | Descrição                          |
| ---------- | ---------------------------------- |
| `arquivos` | Arquivos SVG ou imagem ao iniciar. |

---

## Opções

| Opção               | Descrição                                     |
| ------------------- | --------------------------------------------- |
| `--version`         | Imprimir versão e sair.                       |
| `-h`, `--help`      | Mostrar ajuda e sair.                         |
| `--loglevel NÍVEL`  | Nível de registro. Padrão: `INFO`.            |
| `--config DIR`      | Diretório de configuração personalizado.      |
| `--exit`            | Sair após importação.                         |
| `--vector`          | Forçar importação como vetores diretos.       |
| `--trace`           | Forçar importação por rastreamento de bitmap. |
| `--script SCRIPT`   | Script de inicialização precoce.              |
| `--uiscript SCRIPT` | Script de UI (pós-carregamento).              |

---

## Exemplos

### Abrir um arquivo

```bash
rayforge meuprojeto.ryp
```

### Abrir vários arquivos

```bash
rayforge peca1.svg logo.png design.ryp
```

### Importar com rastreamento

```bash
rayforge --trace foto.png
```

### Executar script precoce e sair

```bash
rayforge --exit --script registrar.py \
    meuprojeto.ryp
```

### Script de UI (automação)

```bash
rayforge --exit --uiscript screenshot.py \
    meuprojeto.ryp
```

### Processamento em lote

```bash
rayforge --exit --vector entrada.svg
```

---

## Scripts precoces (`--script`)

A flag `--script` executa um script Python **sincronicamente
durante a inicialização**, antes dos addons serem carregados
e antes da janela principal ser criada. Útil para:

- Registrar plugins com o gerenciador `pluggy`
- Configurar o contexto da aplicação
- Registrar funções de modelo para caixas de texto
- Definir variáveis de ambiente antes da inicialização

O script tem acesso ao contexto via `get_context()`:

```python
from rayforge.context import get_context

ctx = get_context()
```

### Exemplo: Registrar função de modelo personalizada

```python
"""Registrar função personalizada para modelos de texto.

Executar com: rayforge --script registrar_fn.py
"""
from sketcher.core.template_functions import (
    register_template_function,
)

register_template_function("minha_id", lambda: "PECA-001")
```

Agora `{minha_id()}` funciona em qualquer caixa de texto.

Consulte
[Funções de modelo personalizadas](../features/sketcher/expressions.md#custom-template-functions)
na documentação do sketcher para um tutorial completo.

---

## Scripts de UI (`--uiscript`)

A flag `--uiscript` executa um script Python **após a janela
principal ser completamente carregada**, em uma thread em
segundo plano. Útil para:

- Testes automatizados de UI
- Capturas de tela da aplicação
- Fluxos de trabalho de ponta a ponta

O script pode importar a aplicação e a janela diretamente:

```python
from rayforge.uiscript import app, win
```

O script roda em uma **thread em segundo plano** — cuidado
com a segurança de threads ao acessar widgets GTK
(use `GLib.idle_add` para operações GTK).

### Exemplo: Capturar tela

```python
"""Capturar tela da janela principal."""
from rayforge.uiscript import app, win

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import GLib

def capture():
    surface = win.get_surface()
    if surface:
        surface.write_to_png("/tmp/rayforge_screenshot.png")
    return GLib.SOURCE_REMOVE

GLib.idle_add(capture)
```

---

## Usando ambas as flags

`--script` e `--uiscript` podem ser usados juntos.
O `--script` roda primeiro (sincronicamente), depois a
janela é carregada, e então o `--uiscript` roda:

```bash
rayforge --script setup_precoce.py \
    --uiscript automacao.py \
    meuprojeto.ryp
```

Isso é útil quando você precisa registrar plugins primeiro
e então controlar a UI depois.
