# Deinen Job einrahmen

Erfahre, wie du die Rahmen-Funktion verwendest, um deine Job-Grenzen vorauszuschauen und die korrekte Ausrichtung vor dem Schneiden sicherzustellen.

## Übersicht

Das Einrahmen ermöglicht es dir, die genauen Grenzen deines Laserjobs vorauszuschauen, indem ein Umriss mit dem Laser bei niedriger Leistung oder mit ausgeschaltetem Laser nachgezeichnet wird. Dies hilft, die Positionierung zu verifizieren und kostspielige Fehler zu verhindern.

## Wann Einrahmen verwenden

- **Erstmalige Setups**: Material-Platzierung verifizieren
- **Präzises Positionieren**: Sicherstellen, dass Design in Material-Grenzen passt
- **Mehrere Jobs**: Ausrichtung vor jedem Lauf bestätigen
- **Teure Materialien**: Doppelt prüfen, bevor Schnitte gemacht werden

## Wie einrahmen

### Methode 1: Nur Umriss

Den Job-Grenzrahmen nachzeichnen, ohne den Laser einzuschalten:

1. **Dein Design laden** in Rayforge
2. **Material positionieren** auf dem Laserbett
3. **Auf die Rahmen-Taste klicken** in der Symbolleiste
4. **Den Laserkopf beobachten**, wie er das Begrenzungsrechteck nachzeichnet
5. **Positionierung verifizieren** und Material bei Bedarf anpassen

### Methode 2: Niedrigleistungs-Vorschau

Einige Maschinen unterstützen Niederleistungs-Einrahmen mit sichtbarem Strahl:

1. **Niederleistungs-Modus aktivieren** in den Maschineneinstellungen
2. **Rahmen-Leistung einstellen** (typischerweise 1-5%)
3. **Rahmen-Operation ausführen**
4. **Den Umriss beobachten**, der auf der Materialoberfläche nachgezeichnet wird

:::warning Überprüfe deine Maschine
Nicht alle Laser unterstützen sicher das Niederleistungs-Einrahmen. Konsultiere deine Maschinen-Dokumentation, bevor du diese Funktion verwendest.
:::


## Rahmen-Einstellungen

Rahmen-Verhalten in Einstellungen → Maschine konfigurieren:

- **Rahmen-Geschwindigkeit**: Wie schnell sich der Laserkopf während des Einrahmens bewegt
- **Rahmen-Leistung**: Laserleistung während des Einrahmens (0 für aus, niedrig % für sichtbare Spur)
- **Pause an Ecken**: Kurze Pause an jeder Ecke zur Sichtbarkeit
- **Wiederholungsanzahl**: Anzahl der Male, den Umriss nachzuzeichnen

## Rahmen-Ergebnisse verwenden

Nach dem Einrahmen kannst du:

- **Material-Position anpassen** falls nötig
- **Neu einrahmen** um neue Position zu verifizieren
- **Mit dem Job fortfahren** sobald zufrieden mit der Platzierung

## Tipps für effektives Einrahmen

- **Ecken markieren**: Kleine Klebebandstücke an Ecken als Referenz platzieren
- **Spielraum überprüfen**: Sicherstellen, dass ausreichend Platz um dein Design ist
- **Ausrichtung verifizieren**: Bestätigen, dass Material richtig orientiert ist
- **Schnittbreite berücksichtigen**: Denke daran, dass Schnitte etwas breiter sein werden als Umrisse

## Mit Kamera einrahmen

Wenn deine Maschine Kamera-Unterstützung hat, kannst du:

1. **Kamerabild erfassen** der Material-Platzierung
2. **Design überlagern** auf Kameraansicht
3. **Position virtuell anpassen** vor dem Einrahmen
4. **Einrahmen zur Bestätigung** der physischen Ausrichtung

Siehe [Kamera-Integration](../machine/camera) für Details.

## Fehlerbehebung

**Rahmen stimmt nicht mit Design überein**: Job-Ursprung und Koordinatensystem-Einstellungen überprüfen

**Laser feuert während des Einrahmens**: Rahmen-Leistung deaktivieren oder Maschineneinstellungen überprüfen

**Rahmen zu schnell zum Sehen**: Rahmen-Geschwindigkeit in Einstellungen reduzieren

**Kopf erreicht Ecken nicht**: Verifiziere, dass Design innerhalb des Maschinen-Arbeitsbereichs ist

## Sicherheitshinweise

- **Maschine niemals unbeaufsichtigt lassen** während des Einrahmens
- **Verifiziere, dass Laser aus ist** bei Verwendung von Nullleistungs-Einrahmen
- **Hände frei halten** vom Laserkopf-Pfad
- **Auf Hindernisse achten**, die die Bewegung beeinträchtigen könnten

## Verwandte Themen

- [Kamera-Integration](../machine/camera)
- [Schnellstart-Anleitung](../getting-started/quick-start)
