# No-Go-Zonen

No-Go-Zonen definieren eingeschränkte Bereiche auf der Arbeitsfläche, die der
Laser nicht betreten sollte. Vor dem Ausführen oder Exportieren eines Auftrags
prüft Rayforge, ob Werkzeugwege eine aktivierte No-Go-Zone betreten, und warnt
dich, wenn eine Kollision erkannt wird.

![No-Go-Zonen](/screenshots/machine-nogo-zones.png)

## Eine No-Go-Zone hinzufügen

Öffne **Einstellungen → Maschine** und navigiere zur Seite **No-Go-Zonen**.
Klicke auf die Hinzufügen-Schaltfläche, um eine neue Zone zu erstellen, und
wähle dann ihre Form und Position.

Jede Zone hat die folgenden Einstellungen:

- **Name**: Eine beschreibende Bezeichnung für die Zone
- **Aktiviert**: Die Zone ein- oder ausschalten, ohne sie zu löschen
- **Form**: Rechteck, Box oder Zylinder
- **Position (X, Y, Z)**: Wo die Zone auf der Arbeitsfläche platziert ist
- **Abmessungen**: Breite, Höhe und Tiefe (oder Radius bei Zylindern)

## Kollisionswarnungen

Wenn du einen Auftrag ausführst oder exportierst, prüft Rayforge alle
Werkzeugwege gegen aktivierte No-Go-Zonen. Wenn ein Werkzeugweg durch eine
Zone verläuft, erscheint ein Warndialog mit der Option abzubrechen oder auf
eigenes Risiko fortzufahren.

## Sichtbarkeit

No-Go-Zonen werden sowohl auf der 2D- als auch der 3D-Canvas als
halbdurchsichtige Overlays angezeigt. Verwende die No-Go-Zonen-Umschalttaste
im Canvas-Overlay, um sie ein- oder auszublenden. Die Sichtbarkeitseinstellung
wird zwischen Sitzungen gespeichert.

---

## Verwandte Seiten

- [Hardware-Einstellungen](hardware) - Maschinenabmessungen und Achsenkonfiguration
- [3D-Ansicht](../ui/3d-preview) - 3D-Werkzeugweg-Visualisierung
