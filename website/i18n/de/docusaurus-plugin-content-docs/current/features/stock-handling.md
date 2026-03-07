# Rohmaterial-Verwaltung

Rohmaterial in Rayforge repräsentiert das physische Material, das du schneiden oder gravieren wirst. Rohmaterial ist ein **dokument-weites** Konzept—dein Dokument kann ein oder mehrere Rohmaterial-Elemente haben, die unabhängig von Ebenen existieren.

## Rohmaterial hinzufügen

Rohmaterial repräsentiert das physische Stück Material, mit dem du arbeiten wirst. Um Rohmaterial zu deinem Dokument hinzuzufügen:

1. Im **Rohmaterial**-Panel in der Seitenleiste auf die Schaltfläche **Rohmaterial hinzufügen** klicken
2. Ein neues Rohmaterial-Element wird mit Standardabmessungen erstellt (80% deines Maschinen-Arbeitsbereichs)
3. Das Rohmaterial erscheint als Rechteck im Arbeitsbereich, zentriert auf dem Maschinenbett

### Rohmaterial-Eigenschaften

Jedes Rohmaterial-Element hat die folgenden Eigenschaften:

- **Name**: Ein beschreibender Name zur Identifizierung (automatisch nummeriert als "Rohmaterial 1", "Rohmaterial 2", usw.)
- **Abmessungen**: Breite und Höhe des Rohmaterials
- **Dicke**: Die Materialstärke (optional, aber empfohlen für genaue 3D-Vorschau)
- **Material**: Die Art des Materials (im nächsten Schritt zugewiesen)
- **Sichtbarkeit**: Umschalten, um Rohmaterial im Arbeitsbereich anzuzeigen/verstecken

### Rohmaterial-Elemente verwalten

- **Umbenennen**: Den Rohmaterial-Eigenschaften-Dialog öffnen und das Namensfeld bearbeiten
- **Größe ändern**: Rohmaterial-Element im Arbeitsbereich auswählen und die Eckgriffe zum Skalieren ziehen
- **Bewegen**: Rohmaterial-Element im Arbeitsbereich auswählen und zum Neupositionieren ziehen
- **Löschen**: Auf die Löschen-Taste (Papierkorb-Symbol) neben dem Rohmaterial-Element im Rohmaterial-Panel klicken
- **Eigenschaften bearbeiten**: Auf die Eigenschaften-Taste (Dokument-Symbol) klicken, um den Rohmaterial-Eigenschaften-Dialog zu öffnen
- **Sichtbarkeit umschalten**: Auf die Sichtbarkeit-Taste (Augen-Symbol) klicken, um das Rohmaterial-Element anzuzeigen/verstecken

## Material zuweisen

Nachdem du Rohmaterial definiert hast, kannst du ihm ein Material zuweisen:

1. Im **Rohmaterial**-Panel auf die Eigenschaften-Taste (Dokument-Symbol) beim Rohmaterial-Element klicken
2. Im Rohmaterial-Eigenschaften-Dialog auf die **Auswählen**-Taste neben dem Material-Feld klicken
3. Durch deine Materialbibliotheken browsen und das entsprechende Material auswählen
4. Das Rohmaterial wird aktualisiert, um das visuelle Erscheinungsbild des Materials anzuzeigen

### Material-Eigenschaften

Materialien definieren die visuellen Eigenschaften deines Rohmaterials:

- **Visuelles Erscheinungsbild**: Farbe und Muster zur Visualisierung
- **Kategorie**: Gruppierung (z.B. "Holz", "Acryl", "Metall")
- **Beschreibung**: Zusätzliche Informationen über das Material

Hinweis: Materialeigenschaften sind in Materialbibliotheken definiert und können nicht durch den Rohmaterial-Eigenschaften-Dialog bearbeitet werden. Die Rohmaterial-Eigenschaften ermöglichen es dir nur, einem Rohmaterial-Element ein Material zuzuweisen.

## Werkstücke in Rohmaterial umwandeln

Du kannst jedes Werkstück in ein Rohmaterial-Element umwandeln. Dies ist nützlich, wenn du ein unregelmäßig geformtes Materialstück hast und dessen genaue Umrisse als Rohmaterial-Grenze verwenden möchtest.

So wandelst du ein Werkstück in Rohmaterial um:

1. Rechtsklick auf das Werkstück in der Zeichenfläche oder im Dokument-Panel
2. Wähle **In Rohmaterial umwandeln** aus dem Kontextmenü
3. Das Werkstück wird durch ein neues Rohmaterial-Element mit der gleichen Form und Position ersetzt

Das neue Rohmaterial-Element:

- Verwendet die Geometrie des Werkstücks als Begrenzung
- Übernimmt den Namen des Werkstücks
- Kann wie jedes andere Rohmaterial-Element ein Material zugewiesen bekommen

## Auto-Layout

Die Auto-Layout-Funktion hilft dir, deine Designelemente effizient innerhalb von Rohmaterial-Grenzen anzuordnen:

1. Die Elemente auswählen, die du anordnen möchtest (oder nichts auswählen, um alle Elemente in der aktiven Ebene anzuordnen)
2. Auf die **Anordnen**-Taste in der Symbolleiste klicken und **Auto-Layout (Werkstücke packen)** wählen
3. Rayforge wird die Elemente automatisch anordnen, um die Materialnutzung zu optimieren

### Auto-Layout-Verhalten

Der Auto-Layout-Algorithmus ordnet Elemente innerhalb der sichtbaren Rohmaterial-Elemente in deinem Dokument an:

- **Wenn Rohmaterial-Elemente definiert sind**: Elemente werden innerhalb der Grenzen sichtbarer Rohmaterial-Elemente angeordnet
- **Wenn kein Rohmaterial definiert ist**: Elemente werden über den gesamten Maschinen-Arbeitsbereich angeordnet

Der Algorithmus berücksichtigt:

- **Element-Grenzen**: Respektiert die Abmessungen jedes Designelements
- **Rotation**: Kann Elemente in 90-Grad-Schritten für bessere Passung drehen
- **Abstand**: Hält einen Rand zwischen Elementen ein (Standard 0.5mm)
- **Rohmaterial-Grenzen**: Hält alle Elemente innerhalb der definierten Grenzen

### Manuelle Layout-Alternativen

Wenn du mehr Kontrolle bevorzugst, bietet Rayforge auch manuelle Layout-Werkzeuge:

- **Ausrichtungs-Werkzeuge**: Links, rechts, zentriert, oben, unten ausrichten
- **Verteilungs-Werkzeuge**: Elemente horizontal oder vertikal verteilen
- **Individuelle Positionierung**: Klicken und Elemente ziehen, um sie manuell zu platzieren

## Tipps für effektive Rohmaterial-Verwaltung

1. **Mit genauen Rohmaterial-Abmessungen beginnen** - Miss dein Material präzise für beste Ergebnisse
2. **Beschreibende Namen verwenden** - Benenne deine Rohmaterial-Elemente klar (z.B. "Birken-Sperrholz 3mm")
3. **Material-Dicke einstellen** - Dies kann für zukünftige Berechnungen und Referenz nützlich sein
4. **Materialien früh zuweisen** - Dies stellt korrekte visuelle Darstellung von Anfang an sicher
5. **Unregelmäßiges Rohmaterial für Reststücke verwenden** - Wandle Werkstücke in Rohmaterial um, wenn du Restmaterial mit benutzerdefinierten Formen verwendest
6. **Passung vor dem Schneiden überprüfen** - Die 2D-Ansicht verwenden, um zu verifizieren, dass alles auf dein Rohmaterial passt

## Fehlerbehebung

### Auto-Layout funktioniert nicht wie erwartet

- Sicherstellen, dass mindestens ein Rohmaterial-Element sichtbar ist
- Sicherstellen, dass Elemente nicht gruppiert sind (zuerst Gruppierung aufheben)
- Versuchen, die Anzahl der gleichzeitig ausgewählten Elemente zu reduzieren
- Verifizieren, dass Elemente in die Rohmaterial-Grenzen passen
