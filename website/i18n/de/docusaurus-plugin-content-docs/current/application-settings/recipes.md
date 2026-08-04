# Rezepte und Einstellungen

![Rezept-Einstellungen](/screenshots/application-recipes.png)

Rayforge bietet ein leistungsstarkes Rezept-System, mit dem du konsistente Einstellungen über deine Laserschneideprojekte hinweg erstellen, verwalten und anwenden kannst. Diese Anleitung behandelt die komplette User Journey von der Erstellung von Rezepten in den allgemeinen Einstellungen bis zum Anwenden auf Operationen und Verwalten von Einstellungen auf Schritt-Ebene.

## Übersicht

Das Rezept-System besteht aus drei Hauptkomponenten:

1. **Rezept-Verwaltung**: Wiederverwendbare Einstellungs-Presets erstellen und verwalten
2. **Rohmaterial-Verwaltung**: Materialeigenschaften und Dicke definieren
3. **Schritt-Einstellungen**: Einstellungen auf einzelne Operationen anwenden und feinabstimmen

## Rezept-Verwaltung

### Rezepte erstellen

Rezepte sind benannte Presets, die alle Einstellungen für spezifische Operationen enthalten. Du kannst Rezepte über die Haupteinstellungs-Schnittstelle erstellen:

#### 1. Rezept-Manager aufrufen

Menü: Bearbeiten → Einstellungen, dann Rezepte auswählen

#### 2. Neues Rezept erstellen

Klicke auf "Neues Rezept hinzufügen", um den Rezept-Editor-Dialog zu öffnen.

**Register "Allgemein"** - Rezeptname und Beschreibung festlegen:

![Rezept-Editor - Register Allgemein](/screenshots/recipe-editor-general.png)

Basisinformationen ausfüllen:

- **Name**: Beschreibender Name (z.B. "3mm Sperrholz Schnitt")
- **Beschreibung**: Optionale detaillierte Beschreibung

#### 3. Anwendbarkeits-Kriterien definieren

**Register "Anwendbarkeit"** - Definieren, wann dieses Rezept vorgeschlagen werden soll:

![Rezept-Editor - Register Anwendbarkeit](/screenshots/recipe-editor-applicability.png)

Alle Kriterien sind optional - lasse jedes Feld auf seinem Wert "Beliebig", um auf alles zu passen:

- **Maschine**: Spezifische Maschine wählen oder auf "Beliebig" lassen
- **Aufgabentyp**: Die Operationskategorie auswählen, für die dieses Rezept gilt
  (Schnitt, Gravur, usw.), oder auf "Beliebig" lassen, um für alle Aufgabentypen zu gelten
- **Schritt-Typ**: Das Rezept auf einen bestimmten Operationstyp einschränken
  (z.B. "Kontur" oder "Raster"). Die Liste wird auf die Schritt-Typen gefiltert,
  die den ausgewählten Aufgabentyp unterstützen. Auf "Beliebiger Typ" lassen, um
  jeden Schritt-Typ innerhalb der Aufgabe abzudecken
- **Material**: Materialtyp auswählen oder für jedes Material offen lassen
- **Min./Max. Dicke**: Minimale und maximale Dickenwerte festlegen

#### 4. Einstellungen konfigurieren

**Register "Einstellungen"** - Leistung, Geschwindigkeit und andere Parameter anpassen:

![Rezept-Editor - Register Einstellungen](/screenshots/recipe-editor-settings.png)

Die Einstellungs-Register passen sich der Auswahl auf dem Register "Anwendbarkeit" an:

- Wenn das Rezept auf einen bestimmten **Schritt-Typ** abzielt, zeigt der Editor
  zwei Einstellungsseiten: eine Seite "Laser" mit den gemeinsamen Prozesseinstellungen
  (Leistung, Air-Assist, usw.) und eine Seite "Schritt-Einstellungen" mit den
  Attributen, die für diesen Schritt-Typ spezifisch sind (z.B. Schnittseite, Schnittreihenfolge)

![Rezept-Editor - Register Schritt-Einstellungen](/screenshots/recipe-editor-step-settings.png)

- Bei Auswahl nur eines **Aufgabentyps** (mit "Beliebiger Typ" als Schritt-Typ)
  wird eine einzelne Seite "Einstellungen" mit den Prozesseinstellungen für diese Aufgabe angezeigt
- Bei "Beliebig" für beide werden nur die Basis-Bewegungseinstellungen (Schnittgeschwindigkeit
  und Fahrgeschwindigkeit) angezeigt, die allen Schritten gemeinsam sind

### Rezept-Matching-System

Rayforge schlägt automatisch die am besten geeigneten Rezepte vor und wendet sie an basierend auf:

- **Maschinen-Kompatibilität**: Rezepte können maschinenspezifisch sein
- **Laserkopf-Kompatibilität**: Rezepte können einen bestimmten Kopf auf der
  Maschine erzwingen
- **Material-Matching**: Rezepte können bestimmte Materialien ansprechen
- **Dickenbereiche**: Rezepte gelten innerhalb definierter Dikengrenzen
- **Aufgabentyp-Matching**: Rezepte sind an bestimmte Operationskategorien gebunden
- **Schritt-Typ-Matching**: Rezepte können auf einen bestimmten Operationstyp
  abzielen (z.B. nur "Kontur"-Schritte)

Ein Rezept passt nur dann, wenn alle seine Kriterien erfüllt sind. Wenn ein neuer
Schritt erstellt wird, durchsucht Rayforge die Rezeptbibliothek nach passenden
Rezepten und wendet automatisch das beste an. Das System verwendet einen
Spezifitäts-Bewertungsalgorithmus, um die relevantesten Rezepte zu priorisieren:

1. Maschinenspezifische Rezepte werden höher bewertet als generische
2. Laserkopf-spezifische Rezepte werden höher bewertet
3. Material-spezifische Rezepte werden höher bewertet
4. Dicken-spezifische Rezepte werden höher bewertet
5. Schritt-Typ-spezifische Rezepte werden höher bewertet

### Rezepte auf Schritte anwenden

Rezepte werden pro Schritt angewendet. Öffne die Einstellungen eines beliebigen Schritts und finde die Zeile "Rezept" im Abschnitt "Allgemein":

- **Auswählen...**: Öffnet eine filterbare Liste von Rezepten. Nutze das Suchfeld
  oder den Schalter "Nur kompatible Rezepte anzeigen", um die Liste einzugrenzen;
  kompatible Rezepte passen auf den Aufgabentyp, Schritt-Typ und die Maschine des
  Schritts sowie auf die Rohmaterialien im Dokument. Die Auswahl eines Rezepts
  wendet alle seine Einstellungen auf den Schritt an.
- **Speichern als...**: Öffnet den Rezept-Editor, der mit den aktuellen Einstellungen,
  der Maschine, dem Material und der Dicke des Schritts vorausgefüllt ist. Das
  Speichern des neuen Rezepts wendet es sofort auf den Schritt an.
- **Aktualisieren**: Erscheint, wenn die Einstellungen des Schritts vom darauf
  angewendeten Rezept abweichen (z.B. nachdem du einen Wert manuell geändert hast).
  Ein Klick darauf überschreibt das gespeicherte Rezept mit den aktuellen
  Einstellungen des Schritts.

Der Name des aktuell angewendeten Rezepts wird in der Zeile angezeigt. Schritte
ohne angewendetes Rezept sind mit "Manuelle Einstellungen" gekennzeichnet.

---

**Verwandte Themen**:

- [Materialien](materials) - Materialeigenschaften verwalten
- [Farbregeln](color-rules) - SVG-Farben Schritttypen beim Import zuordnen
- [Material-Handhabung](../features/stock-handling.md) - Mit Rohmaterialien arbeiten
- [Maschinen-Setup](../machine/general.md) - Maschinen und Laserköpfe konfigurieren
- [Operationsübersicht](../features/operations/contour.md) - Verschiedene Operationstypen verstehen
