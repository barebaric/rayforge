# G-Code-Dialekt-Unterstützung

Rayforge unterstützt mehrere G-Code-Dialekte, um mit verschiedener Controller-Firmware zu arbeiten.

## Unterstützte Dialekte

Rayforge unterstützt derzeit diese G-Code-Dialekte:

| Dialekt                         | Firmware     | Häufige Verwendung           | Status                           |
| ------------------------------- | ------------ | ---------------------------- | -------------------------------- |
| **GRBL (universell)**           | GRBL 1.1+    | Diodenlaser, Hobby-CNC       | Primär, vollständig unterstützt  |
| **GRBL (keine Z-Achse)**        | GRBL 1.1+    | 2D-Laserschneider ohne Z     | Optimierte Variante              |
| **GRBL Dynamic (Tiefenbewusst)**| GRBL 1.1+    | Tiefenbewusstes Lasergravieren| Empfohlen für dynamische Leistung|
| **GRBL Dynamic (keine Z-Achse)**| GRBL 1.1+    | Tiefenbewusstes Lasergravieren| Optimierte Variante              |
| **Smoothieware**                | Smoothieware | Laserschneider, CNC          | Experimentell                    |
| **Marlin**                      | Marlin 2.0+  | 3D-Drucker mit Laser         | Experimentell                    |

:::note Empfohlene Dialekte
:::

**GRBL (universell)** ist der am besten getestete und empfohlene Dialekt für Standard-Laseranwendungen.

    **GRBL Dynamic (Tiefenbewusst)** wird für tiefenbewusstes Lasergravieren empfohlen, bei dem die Leistung während der Schnitte variiert (z.B. Gravur mit variabler Tiefe).
---

## Einen benutzerdefinierten Dialekt erstellen

Um einen benutzerdefinierten G-Code-Dialekt basierend auf einem integrierten Dialekt zu erstellen:

1. Öffnen Sie **Maschineneinstellungen** → **G-Code-Dialekt**
2. Klicken Sie auf das **Kopier-Symbol** bei einem integrierten Dialekt, um einen neuen benutzerdefinierten Dialekt zu erstellen
3. Bearbeiten Sie die Dialekteinstellungen nach Bedarf
4. Speichern Sie Ihren benutzerdefinierten Dialekt

Benutzerdefinierte Dialekte werden in Ihrem Konfigurationsverzeichnis gespeichert und können geteilt werden.

---

## Verwandte Seiten

- [G-Code exportieren](../files/exporting) - Exporteinstellungen
- [Firmware-Kompatibilität](firmware) - Firmware-Versionen
- [Geräteeinstellungen](../machine/device) - GRBL-Konfiguration
- [Makros & Hooks](../machine/hooks-macros) - Benutzerdefinierte G-Code-Injektion
