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

Siehe den [Installationsleitfaden](../../getting-started/installation#windows-developer) für detaillierte MSYS2-Entwickler-Installationsanweisungen.

### Schnellstart

Entwicklungsaufgaben unter Windows werden über das `run.bat`-Skript verwaltet, das ein Wrapper für die MSYS2-Shell ist.

Nach dem Klonen des Repositorys und Abschluss der MSYS2-Einrichtung können Sie diese Befehle von einer Standard-Windows-Eingabeaufforderung oder PowerShell verwenden:

```batch
.\run.bat setup
```

Dies führt `scripts/win/win_setup.sh` aus, um alle notwendigen System- und Python-Pakete in Ihre MSYS2/MinGW64-Umgebung zu installieren.

### Pre-commit-Hooks (Optional)

Um Ihren Code vor jedem Commit automatisch zu formatieren und zu linten, führen Sie dies aus der MSYS2 MINGW64-Shell aus:

```bash
bash scripts/win/win_setup_dev.sh
```

:::note

Pre-commit-Hooks erfordern die Ausführung von Git-Befehlen innerhalb der MSYS2 MINGW64-Shell, nicht aus PowerShell oder Eingabeaufforderung.

:::

### Nützliche Befehle

Alle Befehle werden über das `run.bat`-Skript ausgeführt:

-   `run app`: Die Anwendung aus dem Quellcode ausführen.
    -   Fügen Sie `--loglevel=DEBUG` für ausführlichere Ausgabe hinzu.
-   `run test`: Die vollständige Test-Suite mit `pytest` ausführen.
-   `run lint`: Alle Linter ausführen (`flake8`, `pyflakes`, `pyright`).
-   `run format`: Code mit `ruff` formatieren und automatisch korrigieren.
-   `run build`: Die finale Windows-Executable (`.exe`) erstellen.

Alternativ können Sie die Skripte direkt aus der MSYS2 MINGW64-Shell ausführen:

-   `bash scripts/win/win_run.sh`: Die Anwendung ausführen.
-   `bash scripts/win/win_test.sh`: Die Test-Suite ausführen.
-   `bash scripts/win/win_lint.sh`: Alle Linter ausführen.
-   `bash scripts/win/win_format.sh`: Code formatieren und automatisch korrigieren.
-   `bash scripts/win/win_build.sh`: Die Windows-Executable erstellen.
