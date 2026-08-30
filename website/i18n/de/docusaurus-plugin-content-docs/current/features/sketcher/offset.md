---
description:
  "Vergrößere oder verkleinere Konturen oder erstelle Schlitze mit dem Versatz-Werkzeug im
  Rayforge-Sketcher."
---

# Kontur versetzen

Das Versatz-Werkzeug (`O+F`) vergrößert oder verkleinert eine ausgewählte Kontur um einen
angegebenen Abstand oder erweitert einen offenen Pfad zu einem Schlitz. Wähle die Entitäten aus, die
eine Kontur bilden (oder verwende Doppelklick, um verbundene Geometrie auszuwählen), drücke dann
`O+F` oder verwende den **Versatz**-Eintrag im Kreismenü.

![Kontur-versetzen-Dialog](/screenshots/addons-sketcher-offset-dialog.webp)

Der Dialog fragt nach dem Versatzabstand und zeigt während der Eingabe eine Live-Vorschau des
Ergebnisses auf der Leinwand:

- **Geschlossene Konturen** wachsen bei einem positiven Abstand und schrumpfen bei einem negativen.
  Ein Versatz, über den hinaus die Kontur kollabieren würde, wird abgelehnt.
- **Offene Pfade** werden zu einer geschlossenen Schlitzkontur der angegebenen Breite mit
  abgerundeten Endkappen.

![Bezier-Kontur](/screenshots/addons-sketcher-offset-before.webp)
![Bezier zu einem Schlitz versetzt](/screenshots/addons-sketcher-offset-after.webp)

Beim Versetzen wird die ausgewählte Kontur durch das Ergebnis ersetzt:

- Einzelne Kreise, Bögen und Ellipsen behalten ihren Entitätstyp und werden direkt aktualisiert,
  sodass sie wie zuvor bearbeitbar und einschränkbar bleiben.
- Ketten verbundener Segmente (einschließlich Beziers) werden durch eine Polygon-Entität ersetzt.
  Das Polygon wird als Ganzes bearbeitet: Ziehe seinen Mittelpunkt, um es zu verschieben, und seinen
  Griffpunkt, um es zu drehen oder gleichmäßig zu skalieren.

Enthält die Auswahl mehrere getrennte Konturen, wird jede in einem einzigen Schritt unabhängig
versetzt.
