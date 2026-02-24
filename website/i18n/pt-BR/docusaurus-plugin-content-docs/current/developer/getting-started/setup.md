# Configuração

Este guia cobre a configuração do seu ambiente de desenvolvimento para o Rayforge.

## Linux

### Pré-requisitos

Veja o [Guia de Instalação](../../getting-started/installation#linux-pixi) para instruções de instalação do Pixi.

### Hooks Pre-commit (Opcional)

Para formatar e verificar automaticamente seu código antes de cada commit, você pode instalar hooks pre-commit:

```bash
pixi run pre-commit-install
```

### Comandos Úteis

Todos os comandos são executados via `pixi run`:

-   `pixi run rayforge`: Executa a aplicação.
    -   Adicione `--loglevel=DEBUG` para saída mais detalhada.
-   `pixi run test`: Executa a suíte de testes completa com `pytest`.
-   `pixi run format`: Formata todo o código usando `ruff`.
-   `pixi run lint`: Executa todos os linters (`flake8`, `pyflakes`, `pyright`).

## Windows

### Pré-requisitos

Veja o [Guia de Instalação](../../getting-started/installation#windows-developer) para instruções detalhadas de configuração de desenvolvimento MSYS2.

### Início Rápido

Tarefas de desenvolvimento no Windows são gerenciadas via script `run.bat`, que é um wrapper para o shell MSYS2.

Após clonar o repositório e completar a configuração do MSYS2, você pode usar estes comandos a partir de um Prompt de Comando do Windows padrão ou PowerShell:

```batch
.\run.bat setup
```

Isso executa `scripts/win/win_setup.sh` para instalar todos os pacotes de sistema e Python necessários no seu ambiente MSYS2/MinGW64.

### Hooks Pre-commit (Opcional)

Para formatar e verificar automaticamente seu código antes de cada commit, execute isso do shell MSYS2 MINGW64:

```bash
bash scripts/win/win_setup_dev.sh
```

:::note

Hooks pre-commit requerem executar comandos git dentro do shell MSYS2 MINGW64, não do PowerShell ou Prompt de Comando.

:::

### Comandos Úteis

Todos os comandos são executados via script `run.bat`:

-   `run app`: Executa a aplicação a partir do código fonte.
    -   Adicione `--loglevel=DEBUG` para saída mais detalhada.
-   `run test`: Executa a suíte de testes completa usando `pytest`.
-   `run lint`: Executa todos os linters (`flake8`, `pyflakes`, `pyright`).
-   `run format`: Formata e corrige código automaticamente usando `ruff`.
-   `run build`: Compila o executável final do Windows (`.exe`).

Alternativamente, você pode executar os scripts diretamente do shell MSYS2 MINGW64:

-   `bash scripts/win/win_run.sh`: Executa a aplicação.
-   `bash scripts/win/win_test.sh`: Executa a suíte de testes.
-   `bash scripts/win/win_lint.sh`: Executa todos os linters.
-   `bash scripts/win/win_format.sh`: Formata e corrige código automaticamente.
-   `bash scripts/win/win_build.sh`: Compila o executável do Windows.
