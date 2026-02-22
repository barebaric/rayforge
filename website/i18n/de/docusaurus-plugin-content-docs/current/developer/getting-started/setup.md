# Einrichtung

Dieser Leitfaden behandelt die Einrichtung Ihrer Entwicklungsumgebung für Rayforge.

## Linux

### Voraussetzungen

Siehe den [Installationsleitfaden](../../getting-started/installation#linux-pixi) für Pixi-Installationsanweisungen.

### Pre-commit-Hooks (Optional)

Um Ihren Code vor jedem Commit automatisch zu formatieren und zu linten, können Sie Pre-commit-Hooks installieren:

```bash
pixi run pre-commit-install
```

### Nützliche Befehle

Alle Befehle werden über `pixi run` ausgeführt:

-   `pixi run rayforge`: Die Anwendung ausführen.
    -   Fügen Sie `--loglevel=DEBUG` für ausführlichere Ausgabe hinzu.
-   `pixi run test`: Die vollständige Test-Suite mit `pytest` ausführen.
-   `pixi run format`: Den gesamten Code mit `ruff` formatieren.
-   `pixi run lint`: Alle Linter ausführen (`flake8`, `pyflakes`, `pyright`).

## Windows

### Voraussetzungen

-   [MSYS2](https://www.msys2.org/) (stellt die MinGW64-Umgebung bereit).
-   [Git for Windows](https://git-scm.com/download/win).

### Installation

Entwicklungsaufgaben unter Windows werden über das `run.bat`-Skript verwaltet, das ein Wrapper für die MSYS2-Shell ist.

Nach dem Klonen des Repositorys führen Sie den Setup-Befehl von einer Standard-Windows-Eingabeaufforderung oder PowerShell aus:

```batch
.\run.bat setup
```

Dies führt `scripts/win/win_setup.sh` aus, um alle notwendigen System- und Python-Pakete in Ihre MSYS2/MinGW64-Umgebung zu installieren.

### Nützliche Befehle

Alle Befehle werden über das `run.bat`-Skript ausgeführt:

-   `run app`: Die Anwendung aus dem Quellcode ausführen.
    -   Fügen Sie `--loglevel=DEBUG` für ausführlichere Ausgabe hinzu.
-   `run test`: Die vollständige Test-Suite mit `pytest` ausführen.
-   `run build`: Die finale Windows-Executable (`.exe`) erstellen.
