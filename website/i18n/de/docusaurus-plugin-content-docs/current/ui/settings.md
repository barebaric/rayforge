# Einstellungen

![Allgemeine Einstellungen](/screenshots/app-settings-general.webp)

Passe Rayforge an deinen Workflow und deine Vorlieben an. Öffne die Einstellungen über **Bearbeiten
→ Einstellungen** oder drücke <kbd>Strg+,</kbd>.

## Allgemein

Die Seite Allgemein enthält anwendungsweite Einstellungen.

### Erscheinungsbild

Wähle zwischen dem **System**-, **Hell**- oder **Dunkel**-Design, um dich an deine Desktop-Umgebung
oder persönliche Vorliebe anzupassen. Du kannst auch **Operationsfarben** konfigurieren und wählen,
ob die Laserfarbe oder die Ebenenfarbe zur visuellen Unterscheidung auf der Arbeitsfläche verwendet
wird.

### Einheiten

Konfiguriere die in der gesamten Anwendung verwendeten Anzeigeeinheiten. Du kannst separate
Einheiten für **Länge** (Millimeter, Zoll usw.), **Geschwindigkeit** (mm/min, mm/sek, Zoll/min usw.)
und **Beschleunigung** (mm/s² usw.) festlegen.

### Verhalten

Standardmäßig werden Operationen nach jeder Änderung automatisch neu berechnet. Wenn du an einem
langsameren Rechner arbeitest oder sehr komplexe Dokumente hast, kannst du **Operationen automatisch
aktualisieren** deaktivieren und die Neuberechnung stattdessen manuell über die
Symbolleisten-Schaltfläche auslösen.

Rayforge kann beim Start **automatisch nach Updates suchen**. Wenn aktiviert, wirst du
benachrichtigt, wenn eine neue Version verfügbar ist.

Du kannst auch das **Startverhalten** konfigurieren — mit einem leeren Arbeitsbereich starten, das
letzte Projekt erneut öffnen oder immer eine bestimmte Projektdatei öffnen. Beachte, dass auf der
Kommandozeile angegebene Dateien diese Einstellungen immer außer Kraft setzen.

### Datenschutz

Rayforge kann anonyme Nutzungsdaten senden, um die Anwendung zu verbessern. Es werden keine
persönlichen Informationen gesammelt. Du kannst **Anonyme Nutzung melden** jederzeit ein- oder
ausschalten. Siehe die
[Nutzungsverfolgung](https://rayforge.org/docs/general-info/usage-tracking)-Seite, um mehr darüber
zu erfahren, welche Daten gesammelt und wie sie verwendet werden.

## Mausgesten

Die Seite Mausgesten erlaubt es dir, die Navigationsgesten der 2D-Arbeitsfläche, der
3D-Arbeitsfläche und des Skizzen-Editors neu zuzuweisen. Jeder Eintrag zeigt die aktuell zugewiesene
Maustastenkombination. Klicke auf einen Eintrag, um eine neue Zuweisung aufzunehmen: Drücke die
gewünschte Maustaste (mit oder ohne gedrückte Zusatztasten) oder benutze das Mausrad, und die
Zuweisung wird sofort übernommen.

- **Ansicht schwenken** — halte die zugewiesene Maustaste gedrückt und bewege die Maus, um zu
  schwenken.
- **Ansicht zoomen** — benutze das Mausrad zum Zoomen.
- **Orbit / Drehung (3D-Arbeitsfläche)** — ziehe, um um die Szene zu kreisen oder um die Z-Achse zu
  drehen.
- **Kontextmenü öffnen** — das Kontextmenü der 2D-Arbeitsfläche oder das Werkzeugmenü des
  Skizzen-Editors.
- **Ansicht zurücksetzen** — passt die Ansicht wieder an. Standardmäßig nicht zugewiesen.

Eine Zuweisung kann mit **Aufheben** entfernt oder mit **Auf Standard zurücksetzen**
wiederhergestellt werden. Das Zuweisen einer Geste, die in derselben Ansicht bereits für eine andere
Aktion verwendet wird, wird abgelehnt, sodass eine Maustastenkombination nie zwei Aktionen
gleichzeitig auslöst. Addons können zusätzliche Gestenkonfigurationen bereitstellen, die als
zusätzliche Abschnitte auf dieser Seite erscheinen.

## Weitere Einstellungen

Der Einstellungsdialog enthält weitere Seiten zur Verwaltung anderer Teile der Anwendung. Jede
verfügt über eine eigene Dokumentation:

- [Maschinen](../application-settings/machines.md) — Maschinen hinzufügen, entfernen und
  konfigurieren
- [Materialien](../application-settings/materials.md) — Materialbibliotheken verwalten
- [Rezepte](../application-settings/recipes.md) — gespeicherte Operationsrezepte verwalten
- [Farbregeln](../application-settings/color-rules.md) — SVG-Farben Schritttypen zuordnen
- [KI-Anbieter](../application-settings/ai-provider.md) — KI-Anbieter für die Nutzung durch Addons
  konfigurieren
- [Addons](../application-settings/addons.md) — Erweiterungs-Addons installieren, aktualisieren und
  entfernen
