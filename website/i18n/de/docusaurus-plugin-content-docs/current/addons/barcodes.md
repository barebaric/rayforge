---
description:
  "QR-Codes, Code 128 und EAN-13-Barcodes in Rayforge gravieren. Der Inhalt kann eine Vorlage sein,
  sodass jedes Teil seinen eigenen Code trägt."
---

# Barcodes

Das Barcodes-Addon fügt QR-Codes und Barcodes als Werkstücke hinzu. Sie werden als Vektorumrisse mit
Füllung gezeichnet, sodass eine Gravur-Operation sie sauber graviert und eine Kontur-Operation sie
trotzdem nachfahren kann.

## Einen Barcode hinzufügen

Wähle **Objekt - Barcode hinzufügen**. In der Mitte des Arbeitsbereichs erscheint ein QR-Code und
ist ausgewählt. Bearbeite ihn in der Gruppe **Barcode** des Eigenschaftenbereichs:

- **Inhalt** — der zu kodierende Text. Drücke `Enter` oder die Übernehmen-Schaltfläche, um den Code
  zu aktualisieren.
- **Typ** — QR-Code, Code 128 oder EAN-13.
- **Fehlerkorrektur** (QR) — wie viel des Codes beschädigt sein darf, damit er noch lesbar ist:
  Niedrig, Mittel, Quartil oder Hoch. Höhere Stufen machen den Code größer.
- **Modulgröße** — die Breite des schmalsten Strichs oder eines QR-Quadrats. Modulgröße und
  Strichhöhe werden in deiner gewählten Längeneinheit angezeigt. Das Werkstück nimmt die Größe des
  Codes an, daher bestimmt diese Einstellung seine Größe.
- **Strichhöhe** (Code 128, EAN-13) — die Höhe der Striche.
- **Text anzeigen** (Code 128, EAN-13) — druckt den Inhalt unter die Striche. EAN-13 verwendet das
  übliche Layout, bei dem die Begrenzungszeichen zwischen die Ziffern hinunterreichen.
- **Invertieren** — graviert stattdessen die hellen Teile, für dunkles Material: Der Code wird hell
  auf einem dunklen Rand. **Ruhezone** legt diesen Rand in Modulen fest und lässt sich nur ändern,
  solange **Invertieren** aktiv ist; standardmäßig ist es das Minimum, das der Typ benötigt.

Lass um einen nicht invertierten Code einen freien Rand: 4 Module bei QR-Codes, 10 bei Code 128 und
9 bei EAN-13.

## Typen

- **QR-Code** enthält beliebigen Text, auch Links.
- **Code 128** enthält druckbare ASCII-Zeichen. Ziffernfolgen werden automatisch paarweise gepackt.
- **EAN-13** nimmt 12 Ziffern, die Prüfziffer wird ergänzt. Bei 13 Ziffern wird die Prüfziffer
  kontrolliert. Das Inhaltsfeld wird rot, wenn der Inhalt nicht zum Typ passt; fahre mit der Maus
  darüber, um den Grund zu sehen.

## Vorlagen

Der Inhalt kann wie ein Textfeld `{...}`-Ausdrücke enthalten, zum Beispiel `{2 * 21}`. Mit
[Variabler Text](../features/variable-text.md) erhält mit `SN-{serial:05d}` jedes Teil seine eigene
Seriennummer, und Array-Kopien zählen weiter.
