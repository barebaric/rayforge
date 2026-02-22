# Tasker: Gerenciamento de Tarefas em Segundo Plano

`tasker` é um módulo para executar tarefas de longa duração em segundo plano em uma aplicação GTK sem congelar a UI. Ele fornece uma API simples e unificada para trabalho limitado por I/O (`asyncio`) e limitado por CPU (`multiprocessing`).

## Conceitos Principais

1. **`task_mgr`**: O proxy singleton global que você usa para iniciar e cancelar todas as tarefas
2. **`Task`**: Um objeto representando um único trabalho em segundo plano. Você o usa para rastrear status
3. **`ExecutionContext` (`context`)**: Um objeto passado como primeiro argumento para sua função em segundo plano. Seu código o usa para reportar progresso, enviar mensagens e verificar cancelamento
4. **`TaskManagerProxy`**: Um proxy thread-safe que encaminha chamadas para o TaskManager real rodando na thread principal

## Início Rápido

Todas as tarefas em segundo plano são gerenciadas pelo `task_mgr` global.

### Executando uma Tarefa Limitada por I/O (ex.: rede, acesso a arquivo)

Use `add_coroutine` para funções `async`. Estas são leves e ideais para tarefas que aguardam I/O.

```python
import asyncio
from rayforge.shared.tasker import task_mgr

# Sua função em segundo plano DEVE aceitar `context` como primeiro argumento.
async def minha_tarefa_io(context, url):
    context.set_message("Baixando...")
    # ... realiza download assíncrono ...
    await asyncio.sleep(2) # Simula trabalho
    context.set_progress(1.0)
    context.set_message("Download concluído!")

# Inicie a tarefa a partir do seu código UI (ex.: clique de botão)
task_mgr.add_coroutine(minha_tarefa_io, "http://example.com", key="downloader")
```

### Executando uma Tarefa Limitada por CPU (ex.: computação pesada)

Use `run_process` para funções regulares. Estas rodam em um processo separado para evitar o GIL e manter a UI responsiva.

```python
import time
from rayforge.shared.tasker import task_mgr

# Uma função regular, não async.
def minha_tarefa_cpu(context, iteracoes):
    context.set_total(iteracoes)
    context.set_message("Calculando...")
    for i in range(iteracoes):
        # ... realiza cálculo pesado ...
        time.sleep(0.1) # Simula trabalho
        context.set_progress(i + 1)
    return "Resultado Final"

# Inicie a tarefa
task_mgr.run_process(minha_tarefa_cpu, 50, key="calculator")
```

### Executando uma Tarefa em Thread

Use `run_thread` para tarefas que devem rodar em uma thread mas não requerem isolamento completo de processo. Isso é útil para tarefas que compartilham memória mas ainda não devem bloquear a UI.

```python
import time
from rayforge.shared.tasker import task_mgr

# Uma função regular que vai rodar em uma thread
def minha_tarefa_thread(context, duracao):
    context.set_message("Trabalhando na thread...")
    time.sleep(duracao) # Simula trabalho
    context.set_progress(1.0)
    return "Tarefa thread concluída"

# Inicie a tarefa em uma thread
task_mgr.run_thread(minha_tarefa_thread, 2, key="thread_worker")
```

## Padrões Essenciais

### Atualizando a UI

Conecte ao sinal `tasks_updated` para reagir a mudanças. O handler será chamado de forma segura na thread principal GTK.

```python
def setup_ui(barra_progresso, rotulo_status):
    # Este handler atualiza a UI com base no progresso geral
    def on_tasks_updated(sender, tasks, progress):
        barra_progresso.set_fraction(progress)
        if tasks:
            rotulo_status.set_text(tasks[-1].get_message() or "Trabalhando...")
        else:
            rotulo_status.set_text("Ocioso")

    task_mgr.tasks_updated.connect(on_tasks_updated)

# Depois na sua UI...
# setup_ui(minha_barra_progresso, meu_rotulo)
```

### Cancelamento

Dê às suas tarefas uma `key` para cancelá-las depois. Sua função em segundo plano deve verificar periodicamente `context.is_cancelled()`.

```python
# Na sua função em segundo plano:
if context.is_cancelled():
    print("Tarefa foi cancelada, parando trabalho.")
    return

# No seu código UI:
task_mgr.cancel_task("calculator")
```

### Tratando Conclusão

Use o callback `when_done` para obter o resultado ou ver se ocorreu um erro.

```python
def on_task_finished(task):
    if task.get_status() == 'completed':
        print(f"Tarefa concluída com resultado: {task.result()}")
    elif task.get_status() == 'failed':
        print(f"Tarefa falhou: {task._task_exception}")

task_mgr.run_process(minha_tarefa_cpu, 10, when_done=on_task_finished)
```

## Referência da API

### `task_mgr` (O Proxy do Gerenciador)

- `add_coroutine(coro, *args, key=None, when_done=None)`: Adiciona uma tarefa baseada em asyncio
- `run_process(func, *args, key=None, when_done=None, when_event=None)`: Executa uma tarefa limitada por CPU em um processo separado
- `run_thread(func, *args, key=None, when_done=None)`: Executa uma tarefa em uma thread (compartilha memória com processo principal)
- `cancel_task(key)`: Cancela uma tarefa em execução pela sua chave
- `tasks_updated` (sinal para atualizações de UI): Emitido quando o status da tarefa muda

### `context` (Dentro da sua função em segundo plano)

- `set_progress(value)`: Reporta progresso atual (ex.: `i + 1`)
- `set_total(total)`: Define o valor máximo para `set_progress`
- `set_message("...")`: Atualiza o texto de status
- `is_cancelled()`: Verifica se você deve parar
- `sub_context(...)`: Cria uma subtarefa para operações de múltiplos estágios
- `send_event("nome", data)`: (Apenas processo) Envia dados personalizados de volta para a UI
- `flush()`: Envia imediatamente quaisquer atualizações pendentes para a UI

## Uso no Rayforge

O tasker é usado em todo o Rayforge para:

- **Processamento de pipeline**: Executando o pipeline do documento em segundo plano
- **Operações de arquivo**: Importando e exportando arquivos sem bloquear a UI
- **Comunicação de dispositivo**: Gerenciando operações de longa duração com cortadores a laser
- **Processamento de imagem**: Realizando rastreamento e processamento de imagem intensivos em CPU

Ao trabalhar com o tasker no Rayforge, sempre certifique-se de que suas funções em segundo plano tratam cancelamento adequadamente e fornecem atualizações de progresso significativas para manter uma experiência de usuário responsiva.
