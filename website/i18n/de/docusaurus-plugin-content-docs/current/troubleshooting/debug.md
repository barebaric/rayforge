# Fehlerbehebung & Probleme melden

Wenn du Probleme mit Rayforge erlebst, besonders beim Verbinden oder Steuern deiner Maschine, sind wir hier um zu helfen. Der beste Weg, Support zu erhalten, ist durch Bereitstellung eines detaillierten Debug-Berichts. Rayforge hat ein eingebautes Werkzeug, das dies einfach macht.

## So erstellst du einen Debug-Bericht

Folge diesen einfachen Schritten, um einen Bericht zu generieren und zu teilen:

#### 1. Den Bericht speichern

Gehe zu **Hilfe → Debug-Protokoll speichern** in der Menüleiste. Dies wird alle notwendigen Diagnoseinformationen in eine einzelne `.zip`-Datei verpacken. Speichere diese Datei an einem einprägsamen Ort, wie deinem Schreibtisch.

#### 2. Ein GitHub-Issue erstellen

Gehe zu unserer [GitHub-Issues-Seite](https://github.com/barebaric/rayforge/issues/new/choose) und erstelle ein neues Issue. Bitte gib einen klaren Titel und eine detaillierte Beschreibung des Problems:

- **Was hast du getan?** (z.B. "Ich habe versucht, nach dem Starten der App mit meinem Laser zu verbinden.")
- **Was hast du erwartet, dass passiert?** (z.B. "Ich habe erwartet, dass es sich erfolgreich verbindet.")
- **Was ist tatsächlich passiert?** (z.B. "Es blieb getrennt und das Protokoll zeigte Timeout-Fehler.")

#### 3. Den Bericht anhängen

**Ziehe und ablegen die `.zip`-Datei**, die du gespeichert hast, in das Beschreibungsfeld des GitHub-Issues. Dies wird sie hochladen und an deinen Bericht anhängen.

## Was ist im Debug-Bericht enthalten?

Die generierte `.zip`-Datei enthält technische Informationen, die uns helfen, das Problem schnell zu diagnostizieren. Sie umfasst:

- **Maschinen- & Anwendungseinstellungen:** Deine gespeicherten Maschinenkonfigurationen und Anwendungseinstellungen, was uns hilft, dein Setup nachzustellen.
- **Kommunikationsprotokolle:** Ein detailliertes Protokoll der Daten, die zwischen Rayforge und deinem Laser gesendet wurden.
- **Systeminformationen:** Dein Betriebssystem und die Versionen von Rayforge und installierten Schlüsselbibliotheken.
- **Anwendungszustand:** Andere interne Informationen, die helfen können, die Quelle eines Fehlers einzugrenzen.

> **Hinweis zum Datenschutz:** Der Bericht **enthält keine** deiner Designdateien (SVGs, DXFs usw.) oder persönlichen Betriebssystemdaten. Er enthält nur Informationen, die direkt mit der Rayforge-Anwendung und ihrer Verbindung zu deinem Laser zusammenhängen.

Danke, dass du uns hilfst, Rayforge zu verbessern
