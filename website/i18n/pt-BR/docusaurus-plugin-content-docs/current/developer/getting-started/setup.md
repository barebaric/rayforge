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

-   [MSYS2](https://www.msys2.org/) (fornece o ambiente MinGW64).
-   [Git for Windows](https://git-scm.com/download/win).

### Instalação

Tarefas de desenvolvimento no Windows são gerenciadas via script `run.bat`, que é um wrapper para o shell MSYS2.

Após clonar o repositório, execute o comando de configuração a partir de um Prompt de Comando do Windows padrão ou PowerShell:

```batch
.\run.bat setup
```

Isso executa `scripts/win/win_setup.sh` para instalar todos os pacotes de sistema e Python necessários no seu ambiente MSYS2/MinGW64.

### Comandos Úteis

Todos os comandos são executados via script `run.bat`:

-   `run app`: Executa a aplicação a partir do código fonte.
    -   Adicione `--loglevel=DEBUG` para saída mais detalhada.
-   `run test`: Executa a suíte de testes completa usando `pytest`.
-   `run build`: Compila o executável final do Windows (`.exe`).
