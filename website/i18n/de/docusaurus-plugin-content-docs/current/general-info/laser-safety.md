# Lasersicherheit

Laserschneid- und Gravurgeräte stellen ernsthafte Sicherheitsrisiken dar, darunter Feuer, giftige Dämpfe und dauerhafte Augenschäden.

:::danger Kritische Sicherheitswarnung
**Befolge immer die Sicherheitsrichtlinien und Betriebsanweisungen deines Laserherstellers.** Diese Seite enthält allgemeine Sicherheitsprinzipien, ersetzt aber keine ordnungsgemäße Schulung und die spezifische Sicherheitsdokumentation deiner Maschine.
:::


## Deine wichtigste Sicherheitsressource

**Lese das Handbuch deines Laserherstellers vollständig**, bevor du Laserausrüstung betreibst. Es enthält:

- Spezifische Sicherheitsanforderungen für deine Maschine
- Erforderliche persönliche Schutzausrüstung (PSA)
- Notfallverfahren
- Wartungs- und Inspektionspläne
- Garantie- und Haftungsinformationen

Rayforge ist Steuerungssoftware – sie kann die physischen Sicherheitsanforderungen deiner Laserhardware nicht außer Kraft setzen.

## Wesentliche Sicherheitsprinzipien

### Laser niemals unbeaufsichtigt lassen

**Wichtigste Regel:** Lass einen laufenden Laser niemals unbeaufsichtigt, auch nicht für wenige Sekunden.

**Warum:** Laser können Materialien sofort entzünden. Eine kleine Flamme kann sich in Sekunden zu einem ernsthaften Feuer entwickeln.

**Immer:**

- In Reichweite des Not-Aus-Schalters bleiben
- Den Schneidevorgang kontinuierlich beobachten
- Feuerlöscher griffbereit halten

### Brandschutz

**Vor jedem Auftrag:**

- Arbeitsbereich von brennbaren Materialien räumen
- Schnittbett von Abfällen befreien
- Feuerlöscher bereithalten (ABC- oder CO2-Typ)
- Not-Aus-Knopf lokalisieren

**Bei Feuer:**

1. Sofort Not-Aus drücken
2. Deckel geschlossen halten, wenn das Feuer klein ist (sauerstoffarm)
3. Feuerlöscher einsetzen, wenn das Feuer weiterbrennt
4. Evakuieren und Notdienste rufen, wenn sich das Feuer ausbreitet

### Belüftung ist obligatorisch

**Jedes Laserschneiden erzeugt giftige Dämpfe.** Eine ordnungsgemäße Belüftung ist nicht optional.

**Anforderungen:**

- Abzugssystem mit Außenabluft (keine Umluft)
- Ausreichender Luftstrom für deine Maschinengröße
- Betrieb während aller Laservorgänge
- Regelmäßiger Filterwechsel (bei Verwendung von Filterung)

**Betreibe niemals ohne ordnungsgemäße Belüftung** – du riskierst ernsthafte Gesundheitsschäden und beschädigst die Optik deines Lasers.

### Verbotene Materialien

**Schneiden oder gravieren Sie niemals diese Materialien:**

| Material                            | Gefahr                                   |
| ----------------------------------- | ---------------------------------------- |
| **PVC / Vinyl**                     | Erzeugt Chlorgas (giftig, ätzend)        |
| **ABS-Kunststoff**                  | Erzeugt Cyanidgas (tödlich)              |
| **Polycarbonat**                    | Giftige Dämpfe, schlechte Ergebnisse     |
| **Fiberglas**                       | Glaspartikel schädigen Lunge und Optik   |
| **Jedes chlorhaltige Material**     | Giftig und ätzend                        |

**Im Zweifel:** Prüfe das Sicherheitsdatenblatt (SDB) oder teste eine winzige Probe mit ausgezeichneter Belüftung.

### Augenschutz

Die meisten geschlossenen Lasersysteme erfordern keine Schutzbrille während des normalen Betriebs **wenn die Gehäuse geschlossen ist und über ordnungsgemße Verriegelungen verfügt.**

**Augenschutz erforderlich bei:**

- Öffnen des Gehäuses während des Betriebs
- Wartungs- oder Justierarbeiten
- Maschine mit Sichtfenster ohne lasersichere Filterung
- Wenn vom Hersteller vorgeschrieben

**Prüfe immer die Anforderungen deines Herstellers** – diese variieren je nach Maschinendesign und Lasertyp.

## Rayforge-Sicherheitsfunktionen

Rayforge bietet Werkzeuge zur sicheren Bedienung:

- **[Simulationsmodus](../features/simulation-mode)** – Aufträge vor dem Ausführen voranschauen, um Probleme zu erkennen
- **[Materialtestraster](../features/operations/material-test-grid)** – Sichere Einstellungen für neue Materialien finden
- **Rahmen-Scan** – Positionierung vor dem Schneiden überprüfen

**Diese Funktionen helfen, Fehler zu vermeiden, ersetzen aber keine sicheren Betriebspraktiken.**

## Checkliste vor dem Betrieb

Vor jedem Auftrag:

- [ ] Auftragsanforderungen gelesen und verstanden
- [ ] Material ist für Laserschneiden sicher
- [ ] Arbeitsbereich von brennbaren Materialien geräumt
- [ ] Belüftungssystem läuft
- [ ] Feuerlöscher zugänglich
- [ ] Not-Aus-Position bestätigt
- [ ] Material flach auf dem Bett befestigt
- [ ] Auftrag im Simulationsmodus vorangeschaut

## Notfallverfahren

### Not-Aus

**Wann zu verwenden:**

- Feuer sichtbar
- Ungewöhnliche Geräusche oder Rauch
- Material bewegt sich gefährlich
- Jede Notfallsituation

**Wie:** Roter Not-Aus-Knopf an der Maschine drücken. Alle Bewegungen und Laserleistung stoppen sofort.

### Brandbekämpfung

**Kleines Feuer (eingedämmt):**

1. Not-Aus
2. Deckel wenn möglich geschlossen halten
3. Bei Bedarf Feuerlöscher einsetzen

**Großes Feuer:**

1. Not-Aus
2. Sofort evakuieren
3. Notdienste rufen
4. Großbrände nicht bekämpfen versuchen

### Medizinische Notfälle

**Augenexposition:**

- Sofort ärztliche Hilfe suchen
- Augen nicht reiben
- Laserspezifikationen zum Arzt mitbringen

**Dampfeinatmung:**

- An die frische Luft bewegen
- Bei anhaltenden Symptomen ärztliche Hilfe suchen
- Bei schweren Fällen Giftnotruf anrufen

## Schulung und Verantwortung

**Vor dem Betrieb:**

- Erforderliche Sicherheitsschulungen absolvieren
- Herstellerhandbuch vollständig lesen
- Notfallverfahren verstehen
- Gefahrenklasse und Anforderungen deines spezifischen Lasers kennen

**Fortlaufend:**

- Sicherheitsverfahren regelmäßig überprüfen
- Sicherheitsausrüstung warten
- Über neue Materialien und Gefahren informiert bleiben
- Ungeübte Benutzer niemals die Ausrüstung bedienen lassen

## Wichtige Sicherheitsregeln

:::warning Kritische Sicherheitsregeln
1. **Laser niemals während des Betriebs unbeaufsichtigt lassen**
2. **Niemals ohne ordnungsgemäße Belüftung betreiben**
3. **Verbotene Materialien niemals schneiden (PVC, Vinyl, ABS usw.)**
4. **Immer Feuerlöscher in Reichweite haben**
5. **Immer die Sicherheitsrichtlinien des Herstellers befolgen**
6. **Wissen, wo sich der Not-Aus befindet**
7. **Aufträge vor dem Ausführen im Simulationsmodus voranschauen**
:::


## Zusätzliche Ressourcen

**Deine Verantwortlichkeiten:**

- Alle Sicherheitsanforderungen des Herstellers befolgen
- Lokale Brandschutzvorschriften und -verordnungen einhalten
- Erforderliche Sicherheitsausrüstung warten
- Ordnungsgemäße Schulung für alle Bediener gewährleisten
- Sicherheitsvorfälle dokumentieren und melden

**Herstellerressourcen:**

- Betriebsanleitung der Maschine
- Sicherheitsdatenblätter
- Technischer Support-Kontakt
- Garantie- und Sicherheitskonformitätsdokumentation

**Regulatorische Standards:**

- ANSI Z136 (Nordamerika)
- EN 60825 (Europa)
- Lokale Arbeitsschutzvorschriften

## Verwandte Seiten

- **[Simulationsmodus](../features/simulation-mode)** – Aufträge sicher voranschauen
- **[Materialtestraster](../features/operations/material-test-grid)** – Sichere Einstellungen finden
- **[G-Code-Grundlagen](gcode-basics)** – Lasersteuerungsbefehle verstehen

---

**Denke daran:** Sicherer Laserbetrieb erfordert ständige Wachsamkeit, ordnungsgemäße Ausrüstung und strikte Einhaltung der Sicherheitsverfahren. Im Zweifel konsultiere die Dokumentation deines Herstellers und gehe auf Nummer sicher.
