---
description:
  "Der Befehlsschritt fügt benutzerdefinierten Maschinencode, eine Zeile pro Befehl, an einer
  exakten Position im Workflow einer Ebene ein. Verwende ihn für Vorpositionierung,
  Vorrichtungssteuerung und maschinenspezifische Befehle."
---

# Befehl

Der Befehlsschritt fügt benutzerdefinierten Maschinencode in den Job ein, und zwar genau an der
Position, die der Schritt im Workflow der Ebene einnimmt. Verwende ihn, um Befehle zu senden, die
die Geometrie-Operationen nicht abdecken: Positionieren des Kopfes, Ein- und Ausschalten von
Luftunterstützung oder anderer Ausrüstung rund um bestimmte Operationen oder das Senden
maschinenspezifischer Codes.

![Befehlsschritt-Einstellungen](/screenshots/step-settings-command-general.webp)

## Übersicht

Der Befehlsschritt:

- Enthält einen mehrzeiligen Block Maschinencode, einen Befehl pro Zeile
- Gibt jede Zeile unverändert an der Position des Schritts aus, wenn der Job kodiert wird
- Läuft einmal pro Ebene an seiner Position im Workflow — er wirkt nicht auf Werkstücke
- Funktioniert auch ohne Werkstück in der Ebene, sodass eine Ebene, die nur einen Befehlsschritt
  enthält, trotzdem einen Job erzeugen kann
- Unterstützt dieselben Pfadvariablen wie Makros (siehe unten)

Der Text wird unaufgelöst im Projekt gespeichert, reist also mit der `.ryp`-Datei und dokumentiert
genau, was wohin gesendet wird.

## Wann den Befehlsschritt verwenden

Verwende den Befehlsschritt für:

- Vorpositionieren des Kopfes (z.B. Anheben von Z, bevor ein Schnitt beginnt)
- Ein- und Ausschalten von Luftunterstützung, Kühlmittel oder anderer Ausrüstung zwischen
  Operationen
- Senden controllerspezifischer Codes rund um einen Job
- Kurzes Pausieren mit einem Dwell zwischen zwei Operationen

**Verwende den Befehlsschritt nicht für:**

- Wiederholen von G-Code an Ebenen- oder Werkstückgrenzen —
  [Makros & Hooks](../../machine/hooks-macros.md) lösen automatisch aus und funktionieren auch auf
  Maschinen ohne G-Code
- Ausführen von Programmen auf dem Computer (das wird in dieser Phase nicht unterstützt)

## Einen Befehlsschritt hinzufügen

1. Öffne den Workflow der Ebene im rechten Panel.
2. Klicke auf die Schaltfläche **Schritt hinzufügen** und wähle **Befehl**.
3. Gib den Maschinencode in das Textfeld ein, einen Befehl pro Zeile. Leere Zeilen werden
   übersprungen.

Der Schritt kann vor, zwischen oder nach anderen Schritten platziert werden und kann mehrfach in
einer Ebene verwendet werden. Seine Position im Workflow ist die Position, die seine Zeilen im
erzeugten Maschinencode einnehmen.

## Pfadvariablen

Wie bei Makros können die Zeilen Pfadvariablen enthalten, die beim Kodieren des Jobs aufgelöst
werden. Unbekannte Variablen bleiben unverändert, und `layer.*`-Variablen werden aufgelöst, selbst
wenn der Schritt mitten in einer Ebene ausgeführt wird.

| Variable           | Beispiel   | Beschreibung                                    |
| ------------------ | ---------- | ----------------------------------------------- |
| `{machine.name}`   | `My Laser` | Name der aktiven Maschine                       |
| `{layer.name}`     | `Layer 1`  | Name der gerade verarbeiteten Ebene             |
| `{job.extents[0]}` | `210.0`    | Ausdehnung des Jobs auf der X-Achse (mm)        |
| `{wcs_offset[0]}`  | `5.0`      | X-Versatz des aktiven Arbeitskoordinatensystems |

Zum Beispiel wird ein Kommentar wie `; schneide {layer.name} auf {machine.name}` beim Kodieren mit
den eingesetzten Namen versehen.

## Maschinenunterstützung

Der Befehlsschritt ist auf jeder Maschine verfügbar, aber die Zeilen werden nur für Treiber
ausgegeben, die Maschinencode verarbeiten. Wenn der Treiber der aktiven Maschine dies nicht tut
(z.B. Ruida), zeigt der Schritt eine Warnung an, und seine Zeilen sind nicht Teil der erzeugten
Ausgabe.
