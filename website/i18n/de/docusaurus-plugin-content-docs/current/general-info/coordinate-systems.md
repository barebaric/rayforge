# Koordinatensysteme

Das Verständnis, wie Rayforge Koordinatensysteme handhabt, ist essentiell, um Ihre Arbeit korrekt zu positionieren.

## Arbeitskoordinatensystem (WCS) vs. Maschinenkoordinaten

Rayforge verwendet zwei Hauptkoordinatensysteme:

### Arbeitskoordinatensystem (WCS)

Das WCS ist das Koordinatensystem Ihres Jobs. Wenn Sie ein Design bei (50, 100) auf der Arbeitsfläche positionieren, sind dies WCS-Koordinaten.

- **Ursprung**: Von Ihnen definiert (Standard ist G54)
- **Zweck**: Design und Job-Positionierung
- **Mehrere Systeme**: G54-G59 verfügbar für verschiedene Setups

### Maschinenkoordinaten

Maschinenkoordinaten sind absolute Positionen relativ zur Home-Position der Maschine.

- **Ursprung**: Maschinen-Home (0,0,0) - von der Hardware festgelegt
- **Zweck**: Physische Positionierung auf dem Bett
- **Fest**: Kann nicht durch Software geändert werden

**Beziehung**: WCS-Offsets definieren, wie Ihre Job-Koordinaten auf Maschinenkoordinaten abgebildet werden. Wenn der G54-Offset (100, 50, 0) ist, dann schneidet Ihr Design bei WCS (0, 0) an Maschinenposition (100, 50).

## Koordinaten in Rayforge konfigurieren

### Den WCS-Ursprung setzen

Um Ihren Job auf der Maschine zu positionieren:

1. **Maschine referenzieren** zuerst (`$H`-Befehl oder Home-Taste)
2. **Den Laserkopf verfahren** zu Ihrem gewünschten Job-Ursprung
3. **WCS-Null setzen** über das Steuerungsfeld:
   - "X nullen" klicken, um aktuelles X als Ursprung zu setzen
   - "Y nullen" klicken, um aktuelles Y als Ursprung zu setzen
4. Ihr Job wird nun von dieser Position starten

### Ein WCS auswählen

Rayforge unterstützt G54-G59 Arbeitskoordinatensysteme:

| System | Anwendungsfall |
|--------|----------------|
| G54 | Standard, primärer Arbeitsbereich |
| G55-G59 | Zusätzliche Vorrichtungspositionen |

Das aktive WCS im Steuerungsfeld auswählen. Jedes System speichert seinen eigenen Offset vom Maschinenursprung.

### Y-Achsen-Richtung

Einige Maschinen haben Y steigend nach unten statt nach oben. Konfigurieren Sie dies in:

**Einstellungen → Maschine → Hardware → Achsen**

Wenn Ihre Jobs vertikal gespiegelt herauskommen, schalten Sie die Y-Achsen-Richtungseinstellung um.

## Häufige Probleme

### Job an falscher Position

- **WCS-Offset überprüfen**: `G10 L20 P1` senden, um G54-Offset anzuzeigen
- **Referenzierung verifizieren**: Maschine muss für konsistente Positionierung referenziert sein
- **Y-Achsen-Richtung überprüfen**: Könnte invertiert sein

### Koordinaten driften zwischen Jobs

- **Immer vor Jobs referenzieren**: Stellt konsistente Referenz her
- **Auf G92-Offsets prüfen**: Mit `G92.1`-Befehl löschen

---

## Verwandte Seiten

- [Arbeitskoordinatensysteme (WCS)](work-coordinate-systems) - WCS in Rayforge verwalten
- [Steuerungsfeld](../ui/control-panel) - Jog-Steuerungen und WCS-Tasten
- [G-Code exportieren](../files/exporting) - Job-Positionierungsoptionen
