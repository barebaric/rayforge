---
description:
  "QR-Codes, Code 128 und EAN-13 in Rayforge gravieren. Barcodes sind wiederverwendbare Assets,
  deren Inhalt eine Vorlage sein kann, sodass jedes Teil seinen eigenen Code trägt."
---

# Barcodes

Das Barcodes-Addon fügt QR-Codes, Code 128 und EAN-13 als wiederverwendbare Assets hinzu. Ein
Barcode wird als Vektorumriss mit Füllung gezeichnet, sodass eine Gravur-Operation ihn sauber
graviert und eine Kontur-Operation ihn nachfahren kann.

## Einen Barcode hinzufügen

Wähle **Objekt - Barcode hinzufügen**. In der Mitte des Arbeitsbereichs erscheint ein QR-Code und
ist ausgewählt. Bearbeite ihn in der Gruppe **Barcode** des Eigenschaftenbereichs:

- **Name** — der Asset-Name im Asset-Browser.
- **Inhalt** — der zu kodierende Text. Drücke `Enter` oder die Übernehmen-Schaltfläche.
- **Typ** — QR-Code, Code 128 oder EAN-13.
- **Fehlerkorrektur** (QR) — wie viel des Codes beschädigt werden darf: Niedrig, Mittel, Quartil
  oder Hoch. Höhere Stufen machen den Code größer.
- **Modulgröße** — die Breite des schmalsten Strichs oder eines QR-Quadrats. Modulgröße und
  Strichhöhe folgen deiner Längeneinheit. Das Werkstück nimmt die Größe des Codes an.
- **Strichhöhe** (Code 128, EAN-13) — die Höhe der Striche.
- **Text anzeigen** (Code 128, EAN-13) — druckt den Inhalt unter die Striche. EAN-13 nutzt das
  übliche Layout, bei dem die Begrenzungszeichen in die Ziffern reichen.
- **Invertieren** — graviert die hellen Teile, für dunkles Material. **Ruhezone** legt den dunklen
  Rand in Modulen fest und ist nur bei aktiver Invertierung änderbar.

## Der Asset-Browser

Jeder Barcode ist auch ein Asset. Öffne den Asset-Browser im unteren Bereich:

- **Hinzufügen** (`+`) erstellt ein Barcode-Asset, ohne es zu platzieren. Nutze **Neues Werkstück
  erstellen**, ziehe es auf die Arbeitsfläche oder doppelklicke die Karte.
- Ein Doppelklick auf ein Barcode-Werkstück wählt es aus und zeigt die Barcode-Eigenschaften.
- Duplizieren, Kopieren, Ausschneiden und Löschen funktionieren wie bei anderen Assets.

Änderungen am Asset aktualisieren alle platzierten Kopien. Nutze Vorlagen, wenn sich die Kopien
unterscheiden sollen.

Lass um einen nicht invertierten Code einen freien Rand: 4 Module bei QR-Codes, 10 bei Code 128 und
9 bei EAN-13.

## Typen

- **QR-Code** enthält beliebigen Text, auch Links.
- **Code 128** enthält druckbare ASCII-Zeichen. Ziffernfolgen werden automatisch paarweise gepackt.
- **EAN-13** nimmt 12 Ziffern, die Prüfziffer wird ergänzt. Bei 13 Ziffern wird sie kontrolliert.
  Das Inhaltsfeld wird rot, wenn der Inhalt nicht passt; fahre mit der Maus darüber.

## Vorlagen

Der Inhalt kann wie ein Textfeld `{...}`-Ausdrücke enthalten, zum Beispiel `{2 * 21}`. Mit dem
[Auftragszähler](../features/job-counter.md) erhält mit `SN-{counter:05d}` jedes Teil seine eigene
Nummer, und Array-Kopien zählen weiter.
