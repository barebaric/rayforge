# G-Code-Dialekt-Unterstützung

Rayforge unterstützt mehrere G-Code-Dialekte, um mit verschiedener Controller-Firmware zu arbeiten.

## Unterstützte Dialekte

Rayforge unterstützt derzeit diese G-Code-Dialekte:

| Dialekt                          | Firmware     | Häufige Verwendung                |
| -------------------------------- | ------------ | --------------------------------- |
| **GRBL (universell)**            | GRBL 1.1+    | Diodenlaser, Hobby-CNC            |
| **GRBL (keine Z-Achse)**         | GRBL 1.1+    | 2D-Laserschneider ohne Z          |
| **GRBL Dynamic (Tiefenbewusst)** | GRBL 1.1+    | Tiefenbewusstes Lasergravieren    |
| **GRBL Dynamic (keine Z-Achse)** | GRBL 1.1+    | Tiefenbewusstes Lasergravieren    |
| **Mach4 (M67 Analog)**           | Mach4        | Hochgeschwindigkeits-Rastergravur |
| **Smoothieware**                 | Smoothieware | Laserschneider, CNC               |
| **Marlin**                       | Marlin 2.0+  | 3D-Drucker mit Laser              |

:::note Empfohlene Dialekte
:::

**GRBL (universell)** ist der am besten getestete und empfohlene Dialekt für Standard-Laseranwendungen.

**GRBL Dynamic (Tiefenbewusst)** wird für tiefenbewusstes Lasergravieren empfohlen, bei dem die Leistung während der Schnitte variiert (z.B. Gravur mit variabler Tiefe).

---

## Mach4 (M67 Analog)

Der **Mach4 (M67 Analog)**-Dialekt ist für Hochgeschwindigkeits-Rastergravur mit Mach4-Controllern konzipiert. Er verwendet den M67-Befehl mit Analogausgabe zur präzisen Laserleistungssteuerung.

### Hauptmerkmale

- **M67 Analogausgabe**: Verwendet `M67 E0 Q<0-255>` für Laserleistung statt Inline-S-Befehle
- **Reduzierter Pufferdruck**: Durch die Trennung von Leistungsbefehlen und Bewegungsbefehlen wird der Controller-Puffer bei Hochgeschwindigkeitsoperationen weniger belastet
- **Hochgeschwindigkeits-Raster**: Optimiert für schnelle Rastergravuroperationen

### Wann verwenden

Verwende diesen Dialekt, wenn:

- Du einen Mach4-Controller mit Analogausgabefähigkeit hast
- Du Hochgeschwindigkeits-Rastergravur benötigst
- Dein Controller Pufferüberlauf bei Standard-Inline-S-Befehlen erlebt

### Befehlsformat

Der Dialekt generiert G-Code wie:

```gcode
M67 E0 Q127  ; Laserleistung auf 50% setzen (127/255)
G1 X100 Y200 F1000  ; Zur Position bewegen
M67 E0 Q0    ; Laser ausschalten
```

---

## Einen benutzerdefinierten Dialekt erstellen

Um einen benutzerdefinierten G-Code-Dialekt basierend auf einem integrierten Dialekt zu erstellen:

1. Öffne **Maschineneinstellungen** → **G-Code-Dialekt**
2. Klicke auf das **Kopier-Symbol** bei einem integrierten Dialekt, um einen neuen benutzerdefinierten Dialekt zu erstellen
3. Bearbeite die Dialekteinstellungen nach Bedarf
4. Speichere deinen benutzerdefinierten Dialekt

Benutzerdefinierte Dialekte werden in deinem Konfigurationsverzeichnis gespeichert und können geteilt werden.

---

## Verwandte Seiten

- [G-Code exportieren](../files/exporting) - Exporteinstellungen
- [Firmware-Kompatibilität](firmware) - Firmware-Versionen
- [Geräteeinstellungen](../machine/device) - GRBL-Konfiguration
- [Makros & Hooks](../machine/hooks-macros) - Benutzerdefinierte G-Code-Injektion
