# Rohmaterial-Workflow

Die Rohmaterial-Verwaltung in Rayforge ist ein sequenzieller Prozess, mit dem du das physische Material, mit dem du arbeiten wirst, definieren, ihm Eigenschaften zuweisen und dann deine Designelemente darauf organisieren kannst. Diese Anleitung führt dich durch den kompletten Workflow vom Hinzufügen von Rohmaterial bis zum automatischen Layout deines Designs.

## 1. Rohmaterial hinzufügen

Rohmaterial repräsentiert das physische Stück Material, das du schneiden oder gravieren wirst. Um Rohmaterial zu deinem Dokument hinzuzufügen:

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

## 2. Material zuweisen

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

## 3. Rohmaterial Ebenen zuweisen

Nachdem du Rohmaterial definiert und Materialien zugewiesen hast, kannst du Ebenen bestimmten Rohmaterial-Elementen zuordnen:

1. Im **Ebenen**-Panel die Ebene lokalisieren, die du Rohmaterial zuweisen möchtest
2. Auf die Rohmaterial-Zuweisungstaste klicken (zeigt standardmäßig "Gesamte Fläche")
3. Aus dem Dropdown-Menü das Rohmaterial-Element auswählen, das du dieser Ebene zuordnen möchtest
4. Der Inhalt dieser Ebene wird nun auf die Grenzen des zugewiesenen Rohmaterials beschränkt

Du kannst auch "Gesamte Fläche" wählen, um den gesamten Maschinen-Arbeitsbereich anstelle eines bestimmten Rohmaterial-Elements zu verwenden.

### Warum Rohmaterial Ebenen zuweisen?

- **Layout-Grenzen**: Bietet Grenzen für den Auto-Layout-Algorithmus, innerhalb zu arbeiten
- **Visuelle Organisation**: Hilft, dein Design zu organisieren, indem Ebenen physischen Materialien zugeordnet werden
- **Material-Visualisierung**: Zeigt das visuelle Erscheinungsbild des zugewiesenen Materials auf dem Rohmaterial

## 4. Auto-Layout

Die Auto-Layout-Funktion hilft dir, deine Designelemente effizient anzuordnen:

1. Die Elemente auswählen, die du anordnen möchtest (oder nichts auswählen, um alle Elemente in der aktiven Ebene anzuordnen)
2. Auf die **Anordnen**-Taste in der Symbolleiste klicken und **Auto-Layout (Werkstücke packen)** wählen
3. Rayforge wird die Elemente automatisch anordnen, um die Materialnutzung zu optimieren

### Auto-Layout-Verhalten

Der Auto-Layout-Algorithmus funktioniert je nach deiner Ebenenkonfiguration unterschiedlich:

- **Wenn ein Rohmaterial-Element der Ebene zugewiesen ist**: Elemente werden innerhalb der Grenzen dieses bestimmten Rohmaterial-Elements angeordnet
- **Wenn "Gesamte Fläche" ausgewählt ist**: Elemente werden über den gesamten Maschinen-Arbeitsbereich angeordnet

Der Algorithmus berücksichtigt:
- **Element-Grenzen**: Respektiert die Abmessungen jedes Designelements
- **Rotation**: Kann Elemente in 90-Grad-Schritten für bessere Passung drehen
- **Abstand**: Hält einen Rand zwischen Elementen ein (Standard 0.5mm)
- **Rohmaterial-Grenzen**: Hält alle Elemente innerhalb der definierten Grenzen

### Manuelle Layout-Alternativen

Wenn du mehr Kontrolle bevorzugen, bietet Rayforge auch manuelle Layout-Werkzeuge:
- **Ausrichtungs-Werkzeuge**: Links, rechts, zentriert, oben, unten ausrichten
- **Verteilungs-Werkzeuge**: Elemente horizontal oder verteilen
- **Individuelle Positionierung**: Klicken und Elemente ziehen, um sie manuell zu platzieren

## Tipps für effektive Rohmaterial-Verwaltung

1. **Mit genauen Rohmaterial-Abmessungen beginnen** - Miss dein Material präzise für beste Ergebnisse
2. **Beschreibende Namen verwenden** - Benenne deine Rohmaterial-Elemente klar (z.B. "Birken-Sperrholz 3mm")
3. **Material-Dicke einstellen** - Dies kann für zukünftige Berechnungen und Referenz nützlich sein
4. **Materialien früh zuweisen** - Dies stellt korrekte visuelle Darstellung von Anfang an sicher
5. **Ebenen zur Organisation verwenden** - Trenne verschiedene Teile deines Designs in Ebenen, bevor du Rohmaterial zuweist
6. **Passung vor dem Schneiden überprüfen** - Die 2D-Ansicht verwenden, um zu verifizieren, dass alles auf dein Rohmaterial passt

## Fehlerbehebung

### Auto-Layout funktioniert nicht wie erwartet
- Überprüfen, ob deine Ebene ein Rohmaterial zugewiesen hat
- Sicherstellen, dass Elemente nicht gruppiert sind (zuerst Gruppierung aufheben)
- Versuchen, die Anzahl der gleichzeitig ausgewählten Elemente zu reduzieren
- Verifizieren, dass Elemente in die Grenzen passen (Rohmaterial oder gesamte Fläche)
