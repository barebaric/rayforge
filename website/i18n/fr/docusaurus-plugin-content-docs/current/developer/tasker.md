# Tasker : Gestion des tâches en arrière-plan

`tasker` est un module pour exécuter des tâches longues en arrière-plan d'une application GTK sans figer l'interface utilisateur. Il fournit une API simple et unifiée pour le travail lié aux E/S (`asyncio`) et le travail lié au CPU (`multiprocessing`).

## Concepts de base

1. **`task_mgr`** : Le proxy singleton global que vous utilisez pour démarrer et annuler toutes les tâches
2. **`Task`** : Un objet représentant une seule tâche en arrière-plan. Vous l'utilisez pour suivre l'état
3. **`ExecutionContext` (`context`)** : Un objet passé comme premier argument à votre fonction en arrière-plan. Votre code l'utilise pour rapporter la progression, envoyer des messages et vérifier l'annulation
4. **`TaskManagerProxy`** : Un proxy thread-safe qui transfère les appels au véritable TaskManager s'exécutant dans le thread principal

## Démarrage rapide

Toutes les tâches en arrière-plan sont gérées par le `task_mgr` global.

### Exécuter une tâche liée aux E/S (par exemple, réseau, accès fichier)

Utilisez `add_coroutine` pour les fonctions `async`. Celles-ci sont légères et idéales pour les tâches qui attendent des E/S.

```python
import asyncio
from rayforge.shared.tasker import task_mgr

# Votre fonction en arrière-plan DOIT accepter `context` comme premier argument.
async def my_io_task(context, url):
    context.set_message("Téléchargement...")
    # ... effectuer le téléchargement asynchrone ...
    await asyncio.sleep(2) # Simuler le travail
    context.set_progress(1.0)
    context.set_message("Téléchargement terminé !")

# Démarrer la tâche depuis votre code UI (par exemple, un clic sur bouton)
task_mgr.add_coroutine(my_io_task, "http://example.com", key="downloader")
```

### Exécuter une tâche liée au CPU (par exemple, calcul lourd)

Utilisez `run_process` pour les fonctions régulières. Celles-ci s'exécutent dans un processus séparé pour éviter le GIL et garder l'interface réactive.

```python
import time
from rayforge.shared.tasker import task_mgr

# Une fonction régulière, pas async.
def my_cpu_task(context, iterations):
    context.set_total(iterations)
    context.set_message("Calcul...")
    for i in range(iterations):
        # ... effectuer un calcul lourd ...
        time.sleep(0.1) # Simuler le travail
        context.set_progress(i + 1)
    return "Résultat final"

# Démarrer la tâche
task_mgr.run_process(my_cpu_task, 50, key="calculator")
```

### Exécuter une tâche liée à un thread

Utilisez `run_thread` pour les tâches qui devraient s'exécuter dans un thread mais ne nécessitent pas l'isolation complète du processus. C'est utile pour les tâches qui partagent la mémoire mais ne devraient toujours pas bloquer l'interface.

```python
import time
from rayforge.shared.tasker import task_mgr

# Une fonction régulière qui s'exécutera dans un thread
def my_thread_task(context, duration):
    context.set_message("Travail dans un thread...")
    time.sleep(duration) # Simuler le travail
    context.set_progress(1.0)
    return "Tâche thread terminée"

# Démarrer la tâche dans un thread
task_mgr.run_thread(my_thread_task, 2, key="thread_worker")
```

## Patterns essentiels

### Mettre à jour l'interface

Connectez-vous au signal `tasks_updated` pour réagir aux changements. Le gestionnaire sera appelé en toute sécurité sur le thread GTK principal.

```python
def setup_ui(progress_bar, status_label):
    # Ce gestionnaire met à jour l'interface en fonction de la progression globale
    def on_tasks_updated(sender, tasks, progress):
        progress_bar.set_fraction(progress)
        if tasks:
            status_label.set_text(tasks[-1].get_message() or "Travail...")
        else:
            status_label.set_text("Inactif")

    task_mgr.tasks_updated.connect(on_tasks_updated)

# Plus tard dans votre UI...
# setup_ui(my_progress_bar, my_label)
```

### Annulation

Donnez une `key` à vos tâches pour les annuler plus tard. Votre fonction en arrière-plan devrait vérifier périodiquement `context.is_cancelled()`.

```python
# Dans votre fonction en arrière-plan :
if context.is_cancelled():
    print("La tâche a été annulée, arrêt du travail.")
    return

# Dans votre code UI :
task_mgr.cancel_task("calculator")
```

### Gérer l'achèvement

Utilisez le rappel `when_done` pour obtenir le résultat ou voir si une erreur s'est produite.

```python
def on_task_finished(task):
    if task.get_status() == 'completed':
        print(f"Tâche terminée avec le résultat : {task.result()}")
    elif task.get_status() == 'failed':
        print(f"La tâche a échoué : {task._task_exception}")

task_mgr.run_process(my_cpu_task, 10, when_done=on_task_finished)
```

## Référence API

### `task_mgr` (Le Proxy Manager)

- `add_coroutine(coro, *args, key=None, when_done=None)` : Ajouter une tâche basée sur asyncio
- `run_process(func, *args, key=None, when_done=None, when_event=None)` : Exécuter une tâche liée au CPU dans un processus séparé
- `run_thread(func, *args, key=None, when_done=None)` : Exécuter une tâche dans un thread (partage la mémoire avec le processus principal)
- `cancel_task(key)` : Annuler une tâche en cours par sa clé
- `tasks_updated` (signal pour les mises à jour UI) : Émis lorsque l'état de la tâche change

### `context` (À l'intérieur de votre fonction en arrière-plan)

- `set_progress(value)` : Rapporter la progression actuelle (par exemple, `i + 1`)
- `set_total(total)` : Définir la valeur max pour `set_progress`
- `set_message("...")` : Mettre à jour le texte d'état
- `is_cancelled()` : Vérifier si vous devez arrêter
- `sub_context(...)` : Créer une sous-tâche pour des opérations multi-étapes
- `send_event("name", data)` : (Processus uniquement) Envoyer des données personnalisées à l'UI
- `flush()` : Envoyer immédiatement toute mise à jour en attente à l'UI

## Utilisation dans Rayforge

Le tasker est utilisé dans tout Rayforge pour :

- **Traitement du pipeline** : Exécuter le pipeline de document en arrière-plan
- **Opérations sur les fichiers** : Importer et exporter des fichiers sans bloquer l'interface
- **Communication avec l'appareil** : Gérer les opérations longue durée avec les découpeurs laser
- **Traitement d'image** : Effectuer le tracé et le traitement d'image intensifs en CPU

Lorsque vous travaillez avec le tasker dans Rayforge, assurez-vous toujours que vos fonctions en arrière-plan gèrent correctement l'annulation et fournissent des mises à jour de progression significatives pour maintenir une expérience utilisateur réactive.
