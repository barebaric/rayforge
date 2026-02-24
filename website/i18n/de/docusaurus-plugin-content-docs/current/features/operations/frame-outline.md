# Rahmen-Umriss

Rahmen-Umriss erzeugt einen einfachen rechteckigen Schneidepfad um dein gesamtes Design. Es ist der schnellste Weg, einen sauberen Rahmen hinzuzufügen oder deine Arbeit vom Materialblatt zu schneiden.

## Übersicht

Rahmen-Umriss-Operationen:

- Erstellen eine rechteckige Grenze um alle Inhalte
- Fügen konfigurierbaren Offset/Rand vom Design hinzu
- Unterstützen Schnittbreiten-Kompensation für genaue Größen
- Arbeiten mit jeder Kombination von Objekten auf der Arbeitsfläche

![Rahmen-Umriss Schritt-Einstellungen](/screenshots/step-settings-frame-outline-general.png)

## Wann Rahmen-Umriss verwenden

Verwende Rahmen-Umriss für:

- Hinzufügen eines dekorativen Rahmens um dein Design
- Schneiden deiner Arbeit frei vom Materialblatt
- Erstellen einer einfachen rechteckigen Grenze
- Schnelles Einrahmen ohne komplexe Pfadberechnungen

**Verwende Rahmen-Umriss nicht für:**

- Unregelmäßige Formen um mehrere Objekte (verwende stattdessen [Schrumpfumhüllung](shrink-wrap))
- Schneiden individueller Teile (verwende stattdessen [Kontur](contour))
- Folgen der genauen Form deines Designs

## Eine Rahmen-Umriss-Operation erstellen

### Schritt 1: dein Design anordnen

1. Alle Objekte auf der Arbeitsfläche platzieren
2. Positioniere sie relativ zum Rahmen
3. Der Rahmen wird um den Begrenzungsrahmen aller Inhalte berechnet

### Schritt 2: Rahmen-Umriss-Operation hinzufügen

- **Menü:** Operationen → Rahmen-Umriss hinzufügen
- **Rechtsklick:** Kontextmenü → Operation hinzufügen → Rahmen-Umriss

### Schritt 3: Einstellungen konfigurieren

Konfiguriere die Rahmen-Parameter:

- **Leistung & Geschwindigkeit:** An die Schneideanforderungen deines Materials anpassen
- **Offset:** Distanz von Inhalt-Kante zu Rahmen
- **Pfad-Offset:** Innen-, Außen- oder Mittellinien-Schneiden

## Haupt-Einstellungen

### Leistung & Geschwindigkeit

**Leistung (%):**

- Laserintensität zum Schneiden des Rahmens
- An die Schneideanforderungen deines Materials anpassen

**Geschwindigkeit (mm/min):**

- Wie schnell sich der Laser bewegt
- Langsamer für dickere Materialien

**Durchgänge:**

- Anzahl der Male, den Rahmen zu schneiden
- Normalerweise 1-2 Durchgänge
- Durchgänge für dickere Materialien hinzufügen

### Offset-Distanz

**Offset (mm):**

- Distanz vom Begrenzungsrahmen des Designs zum Rahmen
- Erzeugt einen Rand/Bord um deine Arbeit

**Typische Werte:**

- **0mm:** Rahmen berührt Design-Kante
- **2-5mm:** Kleiner Rand für sauberes Aussehen
- **10mm+:** Großer Rahmen für Montage oder Handhabung

### Pfad-Offset (Schnitt-Seite)

Steuert, wo der Laser relativ zum Rahmenpfad schneidet:

| Schnitt-Seite   | Beschreibung                | Verwendung für                       |
| --------------- | --------------------------- | ------------------------------------ |
| **Mittellinie** | Schneidet direkt auf dem Pfad | Standardschneiden                    |
| **Außen**       | Schneidet außerhalb des Rahmenpfads | Rahmen etwas größer machen          |
| **Innen**       | Schneidet innerhalb des Rahmenpfads | Rahmen etwas kleiner machen         |

### Schnittbreiten-Kompensation

Rahmen-Umriss unterstützt Schnittbreiten-Kompensation:

- Passt automatisch für Laserstrahl-Breite an
- Stellt genaue Endabmessungen sicher
- Verwendet den Schnittbreiten-Wert aus deinen Laserkopf-Einstellungen

## Nachbearbeitungsoptionen

![Rahmen-Umriss Nachbearbeitungseinstellungen](/screenshots/step-settings-frame-outline-post.png)

### Mehrfach-Durchgang

Den Rahmen mehrmals schneiden:

- **Durchgänge:** Anzahl der Wiederholungen
- **Z-Abstieg:** Z zwischen Durchgängen absenken (erfordert Z-Achse)
- Nützlich für dicke Materialien

### Halte-Laschen

Laschen hinzufügen, um das eingerahmte Teil befestigt zu halten:

- Verhindert, dass Teile während des Schneidens fallen
- Laschen-Breite, -Höhe und -Abstand konfigurieren
- Siehe [Halte-Laschen](../holding-tabs) für Details

## Anwendungsfälle

### Dekorativer Rahmen

**Szenario:** Einen sauberen rechteckigen Rahmen um eine Plakette oder ein Schild hinzufügen

**Prozess:**

1. Deinen Inhalt gestalten (Text, Logos, usw.)
2. Rahmen-Umriss mit 3-5mm Offset hinzufügen
3. Bei dekorativen Ritz-Einstellungen schneiden (niedrige Leistung)

**Ergebnis:** Professionell aussehendes gerahmtes Teil

### Frei aus Blatt schneiden

**Szenario:** Deine fertige Arbeit vom Materialblatt entfernen

**Prozess:**

1. Alle anderen Operationen abschließen (gravieren, Kontur-Schnitte)
2. Rahmen-Umriss als letzte Operation hinzufügen
3. Offset auf einen kleinen Rand einstellen

**Vorteile:**

- Saubere Trennung vom Blatt
- Konsistente Kantenqualität
- Einfach als letzter Schritt auszuführen

### Chargen-Verarbeitungs-Grenze

**Szenario:** Eine Schneide-Grenze für mehrere verschachtelte Teile erstellen

**Prozess:**

1. Alle Teile auf der Arbeitsfläche anordnen
2. Individuelle Kontur-Operationen für Teile hinzufügen
3. Rahmen-Umriss um alles herum hinzufügen
4. Rahmen schneidet zuletzt (in separater Ebene)

**Reihenfolge:** Gravieren → Teil-Konturen → Rahmen-Umriss

## Tipps & Best Practices

### Ebenen-Reihenfolge

**Beste Praxis:**

- Rahmen-Umriss in eigene Ebene platzieren
- Rahmen als **letzte** Operation ausführen
- Dies stellt sicher, dass alle anderen Arbeiten zuerst abgeschlossen sind

**Warum zuletzt?**

- Material bleibt während anderer Operationen befestigt
- Verhindert, dass Teile sich verschieben
- Saubereres Endergebnis

### Offset-Auswahl

**Offset wählen:**

- **0-2mm:** Enge Passung, minimaler Materialabfall
- **3-5mm:** Standardrand, sieht professionell aus
- **10mm+:** Extra Material für Handhabung/Montage

**Berücksichtigen:**

- Endverwendung des Stücks
- Ob Kanten sichtbar sein werden
- Materialkosten und Verfügbarkeit

### Qualitätseinstellungen

**Für saubere Rahmen-Schnitte:**

- Luftunterstützung verwenden
- Richtigen Fokus sicherstellen
- Mehr schnellere Durchgänge oft besser als ein langsamer
- Material flach und befestigt halten

## Mit anderen Operationen kombinieren

### Rahmen + Gravur + Kontur

Typischer Workflow für ein fertiges Teil:

1. **Ebene 1:** Details gravieren (Text, Bilder)
2. **Ebene 2:** Individuelle Teile mit Kontur schneiden
3. **Ebene 3:** Rahmen-Umriss (frei schneiden)

**Ausführungsreihenfolge stellt sicher:**

- Gravur geschieht während Material flach und befestigt ist
- Teil-Details werden vor endgültiger Trennung geschnitten
- Rahmen schneidet am Ende alles frei

### Rahmen vs Schrumpfumhüllung

| Funktion          | Rahmen-Umriss               | Schrumpfumhüllung      |
| ----------------- | --------------------------- | ---------------------- |
| **Form**          | Immer rechteckig            | Folgt Objekt-Konturen  |
| **Geschwindigkeit**| Sehr schnell (4 Linien)    | Hängt von Komplexität ab |
| **Anwendungsfall** | Einfache Rahmen, frei schneiden | Effiziente Materialnutzung |
| **Flexibilität**  | Festes Rechteck             | Passt sich Design an   |

**Rahmen-Umriss wählen, wenn:**

- Du einen rechteckigen Rahmen wünschst
- Einfachheit bevorzugt wird
- Frei aus Blatt schneiden

**Schrumpfumhüllung wählen, wenn:**

- Du Materialabfall minimieren möchtest
- Design unregelmäßige Form hat
- Effizienz wichtig ist

## Fehlerbehebung

### Rahmen zu eng/zu locker

- **Anpassen:** Offset-Distanz-Einstellung
- **Überprüfen:** Pfad-Offset (innen/außen/Mittellinie)
- **Verifizieren:** Schnittbreiten-Kompensation ist korrekt

### Rahmen erscheint nicht

- **Überprüfen:** Objekte sind auf der Arbeitsfläche
- **Verifizieren:** Operation ist aktiviert
- **Schauen:** Rahmen kann außerhalb des sichtbaren Bereichs sein (herauszoomen)

### Rahmen schneidet in Design

- **Erhöhen:** Offset-Distanz
- **Überprüfen:** Objekte sind richtig positioniert
- **Verifizieren:** Begrenzungsrahmen-Berechnung umfasst alle Objekte

### Inkonsistente Schnitttiefe

- **Überprüfen:** Material ist flach
- **Verifizieren:** Fokusdistanz ist korrekt
- **Versuchen:** Mehrere Durchgänge bei niedrigerer Leistung

## Technische Details

### Begrenzungsrahmen-Berechnung

Rahmen-Umriss verwendet den kombinierten Begrenzungsrahmen von:

- Alle Werkstücke auf der Arbeitsfläche
- ihre endgültigen transformierten Positionen
- Einschließlich aller angewendeten Rotationen/Skalierungen

### Pfad-Generierung

1. Kombinierten Begrenzungsrahmen berechnen
2. Offset-Distanz anwenden
3. Pfad-Offset anwenden (innen/außen/Mittellinie)
4. Schnittbreiten-Kompensation anwenden
5. Rechteckigen G-Code-Pfad generieren

### G-Code-Beispiel

```gcode
G0 X5 Y5           ; Zum Rahmen-Start bewegen (mit Offset)
M3 S200            ; Laser an bei 80% Leistung
G1 X95 Y5 F500     ; Untere Kante schneiden
G1 X95 Y95         ; Rechte Kante schneiden
G1 X5 Y95          ; Obere Kante schneiden
G1 X5 Y5           ; Linke Kante schneiden (vollständig)
M5                 ; Laser aus
```

## Verwandte Themen

- **[Kontur-Schneiden](contour)** - Individuelle Objektumrisse schneiden
- **[Schrumpfumhüllung](shrink-wrap)** - Effiziente unregelmäßige Grenzen
- **[Halte-Laschen](../holding-tabs)** - Teile während des Schneidens sichern
- **[Mehrschicht-Workflow](../multi-layer)** - Operationen effektiv organisieren
- **[Schnittbreiten-Kompensation](../kerf)** - Dimensionsgenauigkeit verbessern
