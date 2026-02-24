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

Menü: Bearbeiten → Einstellungen → Rezepte

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

- **Aufgabentyp**: Operationstyp auswählen (Schnitt, Gravur, usw.)
- **Maschine**: Spezifische Maschine wählen oder "Beliebige Maschine" lassen
- **Material**: Materialtyp auswählen oder für jedes Material offen lassen
- **Dickenbereich**: Minimale und maximale Dickenwerte festlegen

#### 4. Einstellungen konfigurieren

**Register "Einstellungen"** - Leistung, Geschwindigkeit und andere Parameter anpassen:

![Rezept-Editor - Register Einstellungen](/screenshots/recipe-editor-settings.png)

- Leistung, Geschwindigkeit und andere Parameter anpassen
- Einstellungen passen sich automatisch basierend auf dem gewählten Aufgabentyp an

### Rezept-Matching-System

Rayforge schlägt automatisch die am besten geeigneten Rezepte vor basierend auf:

- **Maschinen-Kompatibilität**: Rezepte können maschinenspezifisch sein
- **Material-Matching**: Rezepte können bestimmte Materialien ansprechen
- **Dickenbereiche**: Rezepte gelten innerhalb definierter Dikengrenzen
- **Fähigkeits-Matching**: Rezepte sind an bestimmte Operationstypen gebunden

Das System verwendet einen Spezifitäts-Bewertungsalgorithmus, um die relevantesten Rezepte zu priorisieren:

1. Maschinenspezifische Rezepte werden höher bewertet als generische
2. Laserkopf-spezifische Rezepte werden höher bewertet
3. Material-spezifische Rezepte werden höher bewertet
4. Dicken-spezifische Rezepte werden höher bewertet

---

**Verwandte Themen**:

- [Materialien](materials) - Materialeigenschaften verwalten
- [Material-Handhabung](../features/stock-handling) - Mit Rohmaterialien arbeiten
- [Maschinen-Setup](../machine/general) - Maschinen und Laserköpfe konfigurieren
- [Operationsübersicht](../features/operations/contour) - Verschiedene Operationstypen verstehen
