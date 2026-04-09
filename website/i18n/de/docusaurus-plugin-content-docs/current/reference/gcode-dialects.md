# G-Code-Dialekt-Unterstützung

Rayforge unterstützt mehrere G-Code-Dialekte, um mit verschiedener Controller-Firmware zu arbeiten.

## Unterstützte Dialekte

Rayforge unterstützt derzeit diese G-Code-Dialekte:

| Dialekt                          | Firmware     | Häufige Verwendung                |
| -------------------------------- | ------------ | --------------------------------- |
| **Grbl (Compat)**                | GRBL 1.1+    | Diodenlaser, Hobby-CNC            |
| **Grbl (Compat, keine Z-Achse)** | GRBL 1.1+    | 2D-Laserschneider ohne Z          |
| **Grbl Raster**                  | GRBL 1.1+    | Optimiert für Rasterarbeit        |
| **GRBL Dynamic (Tiefenbewusst)** | GRBL 1.1+    | Tiefenbewusstes Lasergravieren    |
| **GRBL Dynamic (keine Z-Achse)** | GRBL 1.1+    | Tiefenbewusstes Lasergravieren    |
| **LinuxCNC**                     | LinuxCNC     | Native Bézier-Unterstützung (G5)  |
| **Mach4 (M67 Analog)**           | Mach4        | Hochgeschwindigkeits-Rastergravur |
| **Smoothieware**                 | Smoothieware | Laserschneider, CNC               |
| **Marlin**                       | Marlin 2.0+  | 3D-Drucker mit Laser              |

:::note Empfohlene Dialekte
:::

**Grbl (Compat)** ist der am besten getestete und empfohlene Dialekt für
Standard-Laseranwendungen.

**Grbl Raster** ist optimiert für Rastergravur auf GRBL-Controllern. Er hält
den Laser kontinuierlich im dynamischen Leistungsmodus (M4) und lässt
redundante Vorschubbefehle aus, was zu glatterem und kompakterem G-Code führt.

**GRBL Dynamic (Tiefenbewusst)** wird für tiefenbewusstes Lasergravieren
empfohlen, bei dem die Leistung während der Schnitte variiert (z.B. Gravur
mit variabler Tiefe).

**LinuxCNC** unterstützt native kubische Bézier-Kurven über den G5-Befehl, was
sehr glatten und kompakten G-Code für Kurvenpfade erzeugt. Wenn du diesen
Dialekt verwendest, aktiviere die Option „Bézier-Kurven unterstützen" in den
Erweiterten Maschineneinstellungen, um die G5-Ausgabe zu nutzen.

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

Jeder benutzerdefinierte Dialekt ist eine unabhängige Kopie. Die Änderung eines
Dialekts beeinflusst niemals andere, sodass du frei experimentieren kannst, ohne
dir Sorgen machen zu müssen, eine bestehende Konfiguration zu beschädigen.
Benutzerdefinierte Dialekte werden in deinem Konfigurationsverzeichnis gespeichert
und können geteilt werden.

### Dialekt-Einstellungen

Beim Bearbeiten eines benutzerdefinierten Dialekts bietet die Einstellungsseite
folgende Optionen:

**Kontinuierlicher Lasermodus** hält den Laser während des gesamten Auftrags im
dynamischen Leistungsmodus (M4) aktiv, anstatt M4/M5 zwischen Segmenten
umzuschalten. Dies ist nützlich für Rastergravur, bei der der Laser während der
Scan-Zeilen kontinuierlich eingeschaltet bleiben muss.

**Modaler Vorschub** lässt den Vorschubparameter (F) in Bewegungsbefehlen weg,
wenn er sich seit dem letzten Befehl nicht geändert hat. Dies erzeugt kompakteren
G-Code und reduziert die Datenmenge, die an den Controller gesendet wird.

### Separater Laser-Ein-Befehl zur Fokussierung

Einige Dialekte unterstützen die Konfiguration eines separaten Befehls zum
Einschalten des Lasers bei niedriger Leistung, was für den Fokusmodus nützlich
ist. Dies ermöglicht dir, einen anderen Befehl für das visuelle
"Laserzeiger"-Verhalten zu verwenden als beim tatsächlichen Schneiden oder
Gravieren. Überprüfe die Einstellungsseite deines Dialekts auf diese Option.

---

## Verwandte Seiten

- [G-Code exportieren](../files/exporting) - Exporteinstellungen
- [Firmware-Kompatibilität](firmware) - Firmware-Versionen
- [Geräteeinstellungen](../machine/device) - GRBL-Konfiguration
- [Makros & Hooks](../machine/hooks-macros) - Benutzerdefinierte G-Code-Injektion
