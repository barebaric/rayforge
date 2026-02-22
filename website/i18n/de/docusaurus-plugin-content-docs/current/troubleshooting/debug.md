# Fehlerbehebung & Probleme melden

Wenn Sie Probleme mit Rayforge erleben, besonders beim Verbinden oder Steuern Ihrer Maschine, sind wir hier um zu helfen. Der beste Weg, Support zu erhalten, ist durch Bereitstellung eines detaillierten Debug-Berichts. Rayforge hat ein eingebautes Werkzeug, das dies einfach macht.

## So erstellen Sie einen Debug-Bericht

Folgen Sie diesen einfachen Schritten, um einen Bericht zu generieren und zu teilen:

#### 1. Den Bericht speichern

Gehen Sie zu **Hilfe → Debug-Protokoll speichern** in der Menüleiste. Dies wird alle notwendigen Diagnoseinformationen in eine einzelne `.zip`-Datei verpacken. Speichern Sie diese Datei an einem einprägsamen Ort, wie Ihrem Schreibtisch.

#### 2. Ein GitHub-Issue erstellen

Gehen Sie zu unserer [GitHub-Issues-Seite](https://github.com/barebaric/rayforge/issues/new/choose) und erstellen Sie ein neues Issue. Bitte geben Sie einen klaren Titel und eine detaillierte Beschreibung des Problems:

- **Was haben Sie getan?** (z.B. "Ich habe versucht, nach dem Starten der App mit meinem Laser zu verbinden.")
- **Was haben Sie erwartet, dass passiert?** (z.B. "Ich habe erwartet, dass es sich erfolgreich verbindet.")
- **Was ist tatsächlich passiert?** (z.B. "Es blieb getrennt und das Protokoll zeigte Timeout-Fehler.")

#### 3. Den Bericht anhängen

**Ziehen und ablegen Sie die `.zip`-Datei**, die Sie gespeichert haben, in das Beschreibungsfeld des GitHub-Issues. Dies wird sie hochladen und an Ihren Bericht anhängen.

## Was ist im Debug-Bericht enthalten?

Die generierte `.zip`-Datei enthält technische Informationen, die uns helfen, das Problem schnell zu diagnostizieren. Sie umfasst:

- **Maschinen- & Anwendungseinstellungen:** Ihre gespeicherten Maschinenkonfigurationen und Anwendungseinstellungen, was uns hilft, Ihr Setup nachzustellen.
- **Kommunikationsprotokolle:** Ein detailliertes Protokoll der Daten, die zwischen Rayforge und Ihrem Laser gesendet wurden.
- **Systeminformationen:** Ihr Betriebssystem und die Versionen von Rayforge und installierten Schlüsselbibliotheken.
- **Anwendungszustand:** Andere interne Informationen, die helfen können, die Quelle eines Fehlers einzugrenzen.

> **Hinweis zum Datenschutz:** Der Bericht **enthält keine** Ihrer Designdateien (SVGs, DXFs usw.) oder persönlichen Betriebssystemdaten. Er enthält nur Informationen, die direkt mit der Rayforge-Anwendung und ihrer Verbindung zu Ihrem Laser zusammenhängen.

Danke, dass Sie uns helfen, Rayforge zu verbessern
