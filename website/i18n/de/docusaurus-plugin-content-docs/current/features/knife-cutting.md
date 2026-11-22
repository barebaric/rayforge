---
description:
  "Schneide Vinyl, Karton, Leder und andere Flachmaterialien mit einem Schleppmesser oder einem
  motorisch angetriebenen Tangentialmesser statt mit einem Laser."
---

# Schneiden mit Messer

Rayforge kann neben Lasern auch Messerhalter ansteuern. Zwei integrierte Schritte schneiden
Vektorumrisse mit einem Messer: **Schleppmesser-Schnitt** für einen Schleppmesserhalter und
**Tangentialmesser-Schnitt** für eine motorisch angetriebene Drehklinge. Messerschritte erscheinen
nur auf Maschinen mit einem passenden Messerkopf, sodass eine Maschine Laser- und Messeroperationen
im selben Projekt mischen kann.

## Voraussetzungen

- Eine Maschine mit einem **Schleppmesser**- oder **Tangentialmesser**-Kopf. Füge den Kopf in
  [Maschineneinstellungen → Köpfe](../machine/general.md) über **Neuen Kopf hinzufügen** hinzu.
- Ein G-Code-Controller, der die Maschine ohne Laser bewegen kann. Messerschritte schneiden auf der
  Mittellinie und verwenden keine Laserleistung.
- Das integrierte Drag-Knife-Addon. Es wird mit Rayforge ausgeliefert; falls es deaktiviert wurde,
  aktiviere es unter **Einstellungen → Addons**.

## Einen Messerkopf einrichten

1. Öffne **Maschine → Aktuelle Maschineneinstellungen → Köpfe**.
2. Klicke auf **Neuen Kopf hinzufügen** und wähle **Schleppmesser** oder **Tangentialmesser**.
3. Gib dem Kopf einen Namen. Trage bei einem Schleppmesser den **Klingenversatz** ein: den Abstand
   zwischen Klingenspitze und Drehpunkt der Maschine.
4. Schließe die Einstellungen. Die passenden Messerschritte werden nun beim Hinzufügen einer
   Operation angeboten.

Der Messerkopf ist zugleich eine Maschinenfähigkeit: Operationen, die ein Messer benötigen, sind nur
verfügbar, solange die aktive Maschine einen solchen Kopf hat.

## Schleppmesser-Schnitt

Ein Schleppmesser hat eine Klinge, die hinter dem Drehpunkt nachläuft. Rayforge gleicht den Versatz
automatisch aus, damit die Klinge dem Umriss folgt statt dem Drehpunkt.

1. Wähle die zu schneidenden Umrisse aus.
2. Füge eine Operation **Schleppmesser-Schnitt** hinzu.
3. Wähle den **Messer**-Kopf und stelle **Klingenversatz**, **Sichere Z-Höhe**, **Drehwinkel** und
   die Schnittgeschwindigkeit ein.

![Einstellungen für den Schleppmesser-Schnitt](/screenshots/step-settings-drag-knife-general.webp)

Der Schritt schreibt jede Kontur in den Pfad um, den die Maschine fahren muss: Ein Einlaufbogen
richtet die Klinge auf die erste Schnittrichtung aus, gerade Segmente werden um den Klingenversatz
vorwärts verschoben, Ecken innerhalb des Drehwinkels werden mit abgesenkter Klinge umfahren,
schärfere Ecken heben die Klinge zum Drehen an, und ein Auslaufbogen schneidet über das Pfadende
hinaus.

Stelle den **Klingenversatz** nach der Dokumentation des Halters ein oder miss ihn. Ein zu kleiner
Wert hinterlässt abgerundete Ecken; ein zu großer Wert schießt über sie hinaus. Der **Drehwinkel**
begrenzt, wie weit die Klinge abgesenkt dreht; schärfere Ecken heben das Messer zuerst an, damit die
Klinge nicht seitlich zieht.

## Tangentialmesser-Schnitt

Ein Tangentialmesser hat einen Motor, der die Klinge stets tangential zum Pfad dreht. Der Schritt
hängt jedem Schneidzug die Pfadrichtung als **A**-Achswert an, damit der Controller die Klinge
drehen kann.

1. Wähle die zu schneidenden Umrisse aus.
2. Füge eine Operation **Tangentialmesser-Schnitt** hinzu.
3. Wähle den **Messer**-Kopf und stelle **Winkeltoleranz**, **Radiustoleranz**, **Spindeldrehzahl**,
   **Sichere Z-Höhe** und die Schnittgeschwindigkeit ein.

![Einstellungen für den Tangentialmesser-Schnitt](/screenshots/step-settings-tangential-knife-general.webp)

Richtungsänderungen über der **Winkeltoleranz** oder Bögen enger als die **Radiustoleranz** heben
das Messer auf die sichere Z, drehen es und tauchen wieder ein. Kleinere Änderungen drehen die
Klinge bei abgesenktem Messer an Ort und Stelle. Die **Spindeldrehzahl** wird als M3-Befehl für den
Klingenmotor ausgegeben; Rayforge verwendet nie M4, da der Klingenmotor nur im Uhrzeigersinn läuft.

## Tipps

- Fahre den Job zuerst mit reduzierter Geschwindigkeit, besonders bei dickem oder faserigem
  Material.
- Verwende eine Schneidmatte oder eine geringe Eintauchtiefe, damit die Klinge das Bett nicht ritzt.
- Messerschritte sind normale G-Code-Operationen: Sie lassen sich mit Laserschritten kombinieren,
  umsortieren, duplizieren und wie jede andere Operation in Rezepten speichern.
