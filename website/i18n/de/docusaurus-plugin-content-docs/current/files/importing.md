# Dateien importieren

Rayforge unterstützt das Importieren verschiedener Dateiformate, sowohl Vektor- als auch
Rasterformate. Diese Seite erklärt, wie du Dateien importierst und für beste Ergebnisse optimierst.

## Unterstützte Dateiformate

### Vektorformate

| Format          | Erweiterung       | Importmethode                                        | Am besten für                        |
| --------------- | ----------------- | ---------------------------------------------------- | ------------------------------------ |
| **SVG**         | `.svg`, `.svgz`   | Direkte Vektoren oder Tracing                        | Vektorgrafiken, Logos, Designs       |
| **DXF**         | `.dxf`            | Direkte Vektoren                                     | CAD-Zeichnungen, technische Designs  |
| **CorelDRAW**   | `.cdr`            | Direkte Vektoren                                     | CorelDRAW-Zeichnungen                |
| **PDF**         | `.pdf`            | Direkte Vektoren oder Tracing                        | Dokumente mit Vektorinhalt           |
| **LightBurn**   | `.lbrn`, `.lbrn2` | Direkte Vektoren                                     | LightBurn-Projekte                   |
| **Ruida**       | `.rd`             | Direkte Vektoren                                     | Ruida-Controller-Auftragsdateien     |
| **Illustrator** | `.ai`             | Direkte Vektoren oder Tracing                        | Logos aus Illustrator                |
| **EPS**         | `.eps`, `.ps`     | Direkte Vektoren oder Tracing (benötigt Ghostscript) | Logos, Druckvorlagen                 |
| **HPGL**        | `.plt`, `.hpgl`   | Direkte Vektoren                                     | Plotter- und Schneideplotter-Dateien |
| **G-code**      | `.gcode`, `.nc`   | Direkte Vektoren                                     | CAM- und Laser-G-code                |

### Rasterformate

| Format   | Erweiterung     | Importmethode       | Am besten für                            |
| -------- | --------------- | ------------------- | ---------------------------------------- |
| **PNG**  | `.png`          | Tracing zu Vektoren | Fotos, Bilder mit Transparenz            |
| **JPEG** | `.jpg`, `.jpeg` | Tracing zu Vektoren | Fotos, Bilder mit kontinuierlichen Tönen |
| **BMP**  | `.bmp`          | Tracing zu Vektoren | Einfache Grafiken, Screenshots           |
| **GIF**  | `.gif`          | Tracing zu Vektoren | Einfache Grafiken, Animationen           |
| **TIFF** | `.tif`, `.tiff` | Tracing zu Vektoren | Scans, Druckvorlagen                     |
| **WebP** | `.webp`         | Tracing zu Vektoren | Webbilder                                |

<!-- prettier-ignore-start -->
:::note[Raster-Import]
Alle Rasterbilder werden **traced**, um Vektorpfade zu erstellen, die für Laseroperationen verwendet
werden können. Die Qualität hängt von der Tracing-Konfiguration ab.
:::
<!-- prettier-ignore-end -->

---

## Dateien importieren

### Der Import-Dialog

Rayforge verfügt über einen einheitlichen Import-Dialog, der Live-Vorschau und
Konfigurationsoptionen für alle unterstützten Dateitypen bietet. Der Dialog ermöglicht dir:

- **Import voranschauen**, bevor er zum Dokument hinzugefügt wird
- **Tracing-Einstellungen konfigurieren** für Rasterbilder
- **Importmethode wählen** für SVG-Dateien (direkte Vektoren oder Tracing)
- **Parameter anpassen** wie Schwellenwert, Invertieren und Auto-Schwellenwert

![Import-Dialog](/screenshots/import-dialog.webp)

### Ebenen-Import-Modus

Beim Importieren von Dateien, die mehrere Ebenen enthalten (wie SVG- oder PDF-Dateien mit benannten
Ebenen), bietet der Dialog drei Möglichkeiten, diese zu verarbeiten:

- **Bestehenden zuordnen**: Importierte Ebenen bestehenden Dokumentebenen zuordnen. Die erste Ebene
  aus dem Quelldokument wird in die erste Ebene deines Projekts eingefügt, die zweite in die zweite
  und so weiter. Dies ist die Standardeinstellung und funktioniert gut, wenn dein Dokument bereits
  Ebenen eingerichtet hat.
- **Neue Ebenen**: Für jede importierte Ebene eine neue Ebene im Dokument erstellen. Dies hält
  importierten Inhalt getrennt von bestehender Arbeit.
- **Abflachen**: Alle importierten Ebenen in eine einzige Ebene zusammenführen. Nützlich, wenn du
  alles auf einer Ebene haben möchtest, unabhängig von der ursprünglichen Struktur.

### Methode 1: Dateimenü

1. **Datei Importieren** (oder Strg+I)
2. **Datei auswählen** aus dem Dateiauswahldialog
3. **Importeinstellungen konfigurieren** im Import-Dialog
4. **Vorschau** des Ergebnisses vor dem Import
5. **Auf Import klicken** um zur Canvas und zum Dokumentenbaum hinzuzufügen

### Methode 2: Ziehen und Ablegen

1. **Datei ziehen** aus deinem Dateimanager
2. **Auf Rayforge-Canvas ablegen**
3. **Importeinstellungen konfigurieren** im Import-Dialog
4. **Vorschau** des Ergebnisses vor dem Import
5. **Auf Import klicken** um zur Canvas und zum Dokumentenbaum hinzuzufügen

Rayforge registriert seine Importformate auch beim Betriebssystem. Ein Doppelklick auf eine
unterstützte Datei oder „Mit Rayforge öffnen“ übergibt sie daher direkt an den Importdialog.

### Methode 3: Kommandozeile

```bash
# Rayforge mit einer Datei öffnen
rayforge meineDatei.svg

# Mehrere Dateien
rayforge datei1.svg datei2.dxf
```

### Automatische Größenanpassung beim Import

Beim Importieren von Dateien, die größer als der Arbeitsbereich deiner Maschine sind, wird Rayforge
automatisch:

1. **Herunterskalieren** des importierten Inhalts, um in die Maschinengrenzen zu passen
2. **Seitenverhältnis beibehalten** während der Skalierung
3. **Zentrieren** des skalierten Inhalts im Arbeitsbereich
4. **Benachrichtigung anzeigen** mit der Option, die Größenänderung rückgängig zu machen

Die Größenänderungs-Benachrichtigung erscheint als Toast-Meldung:

- ⚠️ "Importiertes Element war größer als der Arbeitsbereich und wurde herunterskaliert um zu
  passen."
- Enthält eine **"Zurücksetzen"-Schaltfläche** um die automatische Größenänderung rückgängig zu
  machen
- Der Toast bleibt sichtbar bis er geschlossen wird oder die Zurücksetzen-Aktion ausgeführt wird

Dies stellt sicher, dass deine Designs immer in die Fähigkeiten deiner Maschine passen, während du
die Flexibilität hast, die Originalgröße bei Bedarf wiederherzustellen.

### Importierte Dateien synchron halten

Beim Importieren behält Rayforge eine Verknüpfung zur Datei auf der Festplatte. Ändert sich die
Datei später — etwa nach dem Bearbeiten und erneuten Speichern in Inkscape oder deinem CAD-Programm
— bietet Rayforge an, den importierten Inhalt neu zu laden, damit dein Projekt nicht veraltete
Geometrie behält.

![Der Dialog „Importierte Datei geändert“](/screenshots/source-changed-dialog.webp)

Gesteuert wird das über die Option **Importierte Dateien beobachten** in den Einstellungen unter
Allgemein. Ist sie aktiviert (die Voreinstellung), prüft Rayforge die verknüpften Dateien, solange
das Projekt geöffnet ist — auch direkt nach dem Öffnen — und zeigt für jede geänderte Datei einen
Dialog an:

- **Neu laden** ersetzt den importierten Inhalt durch die neue Version. Position, Größe und die
  Einstellungen der betroffenen Werkstücke bleiben erhalten, und das gesamte Neuladen ist ein
  einziger Undo-Schritt.
- **Ignorieren** behält den aktuellen Inhalt. Beim nächsten Ändern der Datei wirst du erneut
  gefragt.
- Aktiviere **Diese Datei immer ohne Nachfrage neu laden**, um diese Datei künftig automatisch neu
  zu laden. So markierte Dateien werden ohne Nachfrage neu geladen.

Haben sich mehrere verknüpfte Dateien gleichzeitig geändert, listet der Dialog alle auf, und du
wählst, welche neu geladen werden sollen.

<!-- prettier-ignore-start -->
:::note[Neu laden passt das Dokument an]
Beim Neuladen wird das Dokument wieder mit der Datei abgeglichen: Elemente, die neu dazugekommen
sind, erscheinen im Projekt, Elemente, die nicht mehr in der Datei sind, werden entfernt, und
Halte-Laschen, die mehr als 1 mm von der neuen Kontur entfernt liegen, werden gelöscht. In Rayforge
bearbeitete Elemente bleiben unverändert. Eine Benachrichtigung fasst zusammen, was das Neuladen
geändert hat.
:::
<!-- prettier-ignore-end -->

### Manuelles Neuladen und Neuverknüpfen

Du kannst jederzeit auch manuell neu laden oder neu verknüpfen — über **Datei → Von Festplatte neu
laden** und **Datei → Quelldatei neu verknüpfen …** oder über das Kontextmenü der Auswahl auf der
Zeichenfläche, der Ebenenliste oder des Asset-Browsers. **Von Festplatte neu laden** liest die
Dateien hinter den ausgewählten Werkstücken erneut ein. Ist die verknüpfte Datei nicht auffindbar,
schlägt Rayforge stattdessen das Neuverknüpfen vor.

**Quelldatei neu verknüpfen …** weist eine importierte Quelle einer anderen Datei zu. Das ist
nützlich, wenn eine Datei verschoben oder umbenannt wurde oder wenn ein Projekt auf einem anderen
Computer geöffnet wird und die verknüpfte Datei dort fehlt. Die Werkstücke behalten Position und
Einstellungen und zeigen den Inhalt der neuen Datei.

---

## SVG-Import

SVG (Scalable Vector Graphics) ist das **empfohlene Format** für Vektor-Designs.

<!-- prettier-ignore-start -->
:::tip[Schrittzuweisung automatisieren]
Du kannst
[Farbregeln](../application-settings/color-rules.md) konfigurieren, um Schritttypen (Kontur, Gravur
usw.) automatisch basierend auf den Farben in deinem SVG zuzuweisen. Wenn Regeln existieren,
standardmäßig der Import-Dialog Formen nach Farbe gruppiert, damit die Regeln greifen.
:::
<!-- prettier-ignore-end -->

### Importoptionen im Dialog

Beim Importieren von SVG bietet der Import-Dialog einen Umschalter, um zwischen zwei Methoden zu
wählen:

#### 1. Originalvektoren verwenden (Empfohlen)

Diese Option ist im Import-Dialog standardmäßig aktiviert.

**Wie es funktioniert:**

- Parst SVG und konvertiert Pfade direkt in Rayforge-Geometrie
- Hohe Wiedergabetreue bei Kurven und Formen
- Behält exakte Vektordaten bei

**Vorteile:**

- Beste Qualität und Präzision
- Bearbeitbare Pfade
- Kleine Dateigröße

**Nachteile:**

- Einige fortgeschrittene SVG-Funktionen werden nicht unterstützt
- Komplexe SVGs können Probleme haben

**Verwenden für:**

- Saubere Vektor-Designs aus Inkscape, Illustrator
- Einfache bis mittlere Komplexität
- Designs ohne fortgeschrittene SVG-Funktionen

#### 2. Bitmap tracen

Deaktiviere "Originalvektoren verwenden", um diese Methode zu nutzen.

**Wie es funktioniert:**

- Rendert SVG zuerst in ein Rasterbild
- Traced das gerenderte Bild, um Vektoren zu erstellen
- Kompatibler, aber weniger präzise

**Vorteile:**

- Verarbeitet komplexe SVG-Funktionen
- Robuste Fallback-Methode
- Unterstützt Effekte und Filter

**Nachteile:**

- Qualitätsverlust durch Rasterung
- Größere Dateigrößen
- Nicht so präzise

**Verwenden für:**

- SVGs, bei denen direkter Import fehlschlägt
- SVGs mit Effekten, Filtern, Farbverläufen
- Wenn direkter Import Fehler produziert

### Ebenenauswahl und nicht gruppierte Geometrie

Mit **Originalvektoren verwenden** zeigt die Liste **Ebenen auswählen** einen Eintrag pro benannter
SVG-Ebene (bzw. pro Farbe bei Gruppierung nach Farben). Formen außerhalb jeder benannten Ebene
werden als zusätzliche Ebene **Nicht gruppiert** angeboten, standardmäßig aktiviert, damit sie wie
jede andere Ebene importiert und geschnitten werden. Das Abwählen einer Ebene schließt ihre
Geometrie sowohl aus der Vorschau als auch aus den erzeugten Werkzeugbahnen aus.

### Live-Vorschau

Der Import-Dialog zeigt eine Live-Vorschau, wie dein SVG importiert wird:

- Vektorpfade werden als blaues Overlay angezeigt
- Für Tracing-Modus wird das Originalbild mit den getraceten Pfaden angezeigt
- Vorschau aktualisiert sich in Echtzeit beim Ändern der Einstellungen

### SVG-Best Practices

**Bereite dein SVG für beste Ergebnisse vor:**

1. **Text in Pfade konvertieren:**
   - Inkscape: `Pfad → Objekt in Pfad`
   - Illustrator: `Typ → Umrisse erstellen`

2. **Komplexe Pfade vereinfachen:**
   - Inkscape: `Pfad → Vereinfachen` (Strg+L)
   - Unnötige Knoten entfernen

3. **Verschachtelte Gruppen auflösen:**
   - Hierarchie wo möglich abflachen
   - `Objekt → Gruppierung aufheben` (Strg+Umschalt+G)

4. **Versteckte Elemente entfernen:**
   - Hilfslinien, Raster, Konstruktionslinien löschen
   - Unsichtbare/transparente Objekte entfernen

5. **Als Plain SVG speichern:**
   - Inkscape: "Plain SVG" oder "Optimiertes SVG"
   - Nicht "Inkscape SVG" (hat zusätzliche Metadaten)

6. **Dokumenteneinheiten prüfen:**
   - Auf mm oder Zoll wie angemessen einstellen
   - Rayforge verwendet intern mm

**Häufige SVG-Funktionen, die möglicherweise nicht importiert werden:**

- Farbverläufe (in einfarbige Füllungen oder Raster konvertieren)
- Filter und Effekte (in Pfade umwandeln)
- Masken und Beschneidungspfade (erweitern/abflachen)
- Eingebettete Rasterbilder (separat exportieren)
- Text (zuerst in Pfade konvertieren)

---

## DXF-Import

DXF (Drawing Exchange Format) ist üblich für CAD-Software.

### DXF-Versionen

Rayforge unterstützt Standard-DXF-Formate:

- **R12/LT2** (empfohlen) - Beste Kompatibilität
- **R13, R14** - Gute Unterstützung
- **R2000+** - Funktioniert normalerweise, aber R12 ist sicherer

**Tipp:** Exportiere als R12/LT2 DXF für maximale Kompatibilität.

### Abflachen der Z-Koordinaten

Manche CAD-Exporteure schreiben DXF-Dateien mit Geometrie auf unterschiedlichen Z-Höhen – zum
Beispiel ein 3D-Modell, das mit den Konturen der Ober- und der Unterseite eines Teils exportiert
wurde. Rayforge ist ein 2D-Programm: Auf einer Maschine mit Z-Achse werden solche Z-Werte zu echten
Z-Bewegungen im Job, und der Kopf kann unter die Materialoberfläche eintauchen und abstürzen.

Enthält eine DXF-Datei Geometrie auf mehr als einer Z-Höhe, bietet der Import-Dialog daher eine
Option **Z-Koordinaten abflachen** (unter „Import-Modus“) an. Sie ist standardmäßig aktiviert und
projiziert die gesamte importierte Geometrie auf die Z=0-Ebene. Du kannst sie deaktivieren, um die
ursprünglichen Z-Höhen beizubehalten – der Dialog warnt dich dann, dass sich der Werkzeugpfad in Z
bewegt, was nur sicher ist, wenn du weißt, dass deine Datei dies benötigt (zum Beispiel, um eine
bestimmte Fokushöhe pro Form nachzubilden).

Bei flachen Zeichnungen wird die Option nicht angezeigt; es gibt nichts abzuflachen.

### Offene Konturen reparieren

CAD-Exporteure schreiben eine Kontur häufig als viele einzelne Linien und Bögen, deren Enden nur
fast aneinanderstoßen. Eine solche Datei wird als Ansammlung offener Fragmente importiert, die sich
nicht versetzen, nicht in Teile zerlegen und nicht als geschlossene Bahn schneiden lassen –
typisches Symptom: **Objekt → Teilen** scheint nichts zu tun.

Bei Exporteuren, die dazu neigen, bietet der Import-Dialog die Option **Offene Konturen reparieren**
an. Sie ist standardmäßig ausgeschaltet und erscheint unter „Kontur-Reparatur“. Wenn sie aktiviert
ist, verbindet Rayforge Segmente, deren Enden innerhalb der Toleranz **Verbindungslücke** liegen,
schließt Konturen, deren eigene Enden danach zusammenfallen, und entfernt Konturen, die doppelt
vorkommen. Bestehende Punkte werden nie verschoben: Jede Verbindung wird mit einer geraden Strecke
überbrückt, sodass eine Form an jeder Verbindungsstelle höchstens um die Toleranz wächst.

Lassen Sie **Verbindungslücke** auf dem Mindestwert, damit Rayforge eine zur Zeichnungsgröße
passende Toleranz wählt. Erhöhen Sie den Wert, wenn nach dem Import noch Konturen offen sind – der
Import-Dialog meldet, wie viele übrig blieben. Bei zu hohen Werten können fremde Konturen fälschlich
verbunden werden.

### DXF-Import-Tipps

**Vor dem Export aus CAD:**

1. **Zeichnung vereinfachen:**
   - Unnötige Ebenen entfernen
   - Bemaßungen und Anmerkungen löschen
   - 3D-Objekte entfernen (2D-Projektion verwenden)

2. **Einheiten prüfen:**
   - Zeichnungseinheiten verifizieren (mm vs. Zoll)
   - Rayforge geht standardmäßig von mm aus

3. **Ebenen abflachen:**
   - Erwäge, nur relevante Ebenen zu exportieren
   - Konstruktions-Ebenen ausblenden oder löschen

4. **Angemessene Präzision verwenden:**
   - Laser-Präzision ist typischerweise 0,1 mm
   - Präzision nicht überspezifizieren

**Nach dem Import:**

- Maßstab prüfen (DXF-Einheiten benötigen möglicherweise Anpassung)
- Verifizieren, dass alle Pfade korrekt importiert wurden
- Unerwünschte Konstruktionselemente löschen

---

## CorelDRAW-Import

CorelDRAW-Dateien (`.cdr`) werden nativ gelesen; CorelDRAW muss dafür nicht installiert sein.
Rayforge liest die Zeichnungen von CorelDRAW 7 bis X7, sowohl als reine Dateien als auch als
ZIP-basierte Pakete späterer Versionen.

**Unterstützte Funktionen:**

- ✓ Kurven, Pfade, Bögen, Ellipsen und abgerundete Rechtecke
- ✓ Text als Konturen importiert
- ✓ Benannte Ebenen, im Importdialog auswählbar
- ✓ Objekt-, Gruppen- und Ebenentransformationen
- ✓ Ausgeblendete Ebenen und Objekte werden ausgelassen

**Einschränkungen:**

- ✗ Nur die erste Zeichnungsseite wird importiert; weitere Seiten werden gemeldet
- ✗ Bitmaps und nicht unterstützte Objekttypen werden als Warnungen gemeldet und ausgelassen
- ✗ Füllungen, Konturen und eingebettete Schriften werden ignoriert; Text wird mit der am besten
  passenden installierten Schrift gezeichnet, sodass Glyphen von CorelDRAW abweichen können
- ✗ CMX-Dateien werden erkannt, können aber nicht gelesen werden

**Wann verwenden:**

- Entwürfe öffnen, die nur als CorelDRAW-Datei vorliegen
- Einen bestehenden CorelDRAW-Workflow nach Rayforge überführen

---

## PDF-Import

PDF-Dateien können Vektorgrafiken, Rasterbilder oder beides enthalten.

### Direkter Vektorimport

Beim Importieren eines PDFs, das Vektorpfade enthält, kann Rayforge diese direkt importieren – genau
wie SVG- oder DXF-Dateien. Das liefert dir saubere, skalierbare Geometrie ohne Qualitätsverlust
durch Rasterung.

Wenn das PDF Ebenen enthält, erkennt Rayforge diese und du kannst auswählen, welche importiert
werden sollen. Jede Ebene wird zu einem separaten Werkstück in deinem Dokument. Dies funktioniert
genauso wie der SVG-Ebenenimport: aktiviere oder deaktiviere einzelne Ebenen im Import-Dialog vor
dem Importieren.

Dies ist besonders nützlich für PDFs, die aus Design-Software wie Illustrator oder Inkscape
exportiert wurden, wo die Vektorpfade sauber und gut organisiert sind.

### Fallback: Rendern und Tracen

Für PDFs, die keine verwendbaren Vektordaten enthalten – gescannte Dokumente, eingebettete Fotos
oder PDFs, bei denen Text nicht in Pfade umgewandelt wurde – kann Rayforge auf das Rendern des PDFs
als Bild und dessen Tracing zurückgreifen. Dies funktioniert genauso wie der Rasterbild-Import.

### PDF-Import-Tipps

**Beste Ergebnisse:**

1. **Vektor-PDFs verwenden**: PDFs, die aus Vektorsoftware erstellt wurden (Illustrator, Inkscape),
   liefern die saubersten Ergebnisse beim direkten Import.

2. **Auf Ebenen prüfen**: Wenn dein PDF Ebenen hat, werden diese im Import-Dialog aufgelistet. Wähle
   nur die Ebenen aus, die du benötigst.

3. **Für Dokumente mit Text**: Als SVG mit in Pfade konvertierten Schriftarten exportieren für beste
   Qualität, oder das Rendern-und-Tracing-Fallback verwenden.

4. **Import-Dialog-Vorschau verwenden**: Schwellenwert- und Invertierungseinstellungen im
   Tracing-Modus anpassen. Die Vorschau zeigt genau, wie das PDF getraced wird.

---

## Ruida-Import

Ruida-Dateien (.rd) sind proprietäre binäre Auftragsdateien, die von Ruida-Controllern in vielen
Laserschneidmaschinen verwendet werden. Diese Dateien enthalten sowohl Vektorgeometrie als auch
Lasereinstellungen, organisiert in Ebenen (Farben).

**Nach dem Import:**

- **Maßstab prüfen** – Verifizieren, dass Abmessungen der erwarteten Größe entsprechen
- **Ebenen überprüfen** – Sicherstellen, dass alle Ebenen korrekt importiert wurden
- **Pfade validieren** – Bestätigen, dass alle Schneidepfade vorhanden sind

### Einschränkungen

- **Nur-Lese-Import** – Ruida-Dateien können nur importiert, nicht exportiert werden
- **Binärformat** – Direktes Bearbeiten von Original-.rd-Dateien nicht unterstützt
- **Proprietäre Funktionen** – Einige fortgeschrittene Ruida-Funktionen werden möglicherweise nicht
  vollständig unterstützt

---

## LightBurn-Import

LightBurn-Dateien (.lbrn / .lbrn2) sind proprietäre Projektdateien, die von der
LightBurn-Laserschneidesoftware verwendet werden. Sie speichern Vektorgeometrie, die in
farbcodierten Ebenen mit konfigurierbaren Laser-Einstellungen organisiert ist.

**Nach dem Import:**

- **Ebenen überprüfen** — LightBurn-Ebenen werden auf Dokumentebenen mit entsprechenden Namen
  abgebildet
- **Laser-Einstellungen prüfen** — Leistungs-, Geschwindigkeits- und Durchgangseinstellungen aus
  LightBurn-Ebenen bleiben erhalten
- **Pfade validieren** — Bestätigen, dass alle Schnittpfade vorhanden und korrekt positioniert sind

### Einschränkungen

- **Schreibgeschützter Import** — LightBurn-Dateien können nur importiert, nicht exportiert werden
- **Binärformat** — Direktes Bearbeiten von Original-.lbrn-Dateien nicht unterstützt
- **Proprietäre Funktionen** — Einige fortgeschrittene LightBurn-Funktionen werden möglicherweise
  nicht vollständig unterstützt

---

## G-code-Import

G-code (`.gcode`, `.ngc`, `.nc`, `.tap`, `.gc`) wird nur als Geometrie importiert. Rayforge liest
G0/G1/G2/G3-Bewegungen, G20/G21-Einheiten und G90/G91-Entfernungsmodi, einschließlich I/J- und
R-Bögen. Eilgänge und Vorschubbewegungen mit Spindeldrehzahl null gelten als Verfahrwege, damit
LightBurn-Füllungen und Overscan keine Schnittlinien werden. Vorschub, Leistung, Spindel und
Kühlmittel werden ignoriert; die Datei kommt als einzelne Ebene an.

**Nach dem Import:**

- **Operationen zuweisen** — Schnitt- oder Gravureinstellungen selbst wählen; sie werden nicht aus
  der Datei gelesen
- **Maßstab prüfen** — Prüfe, ob die Abmessungen zum Koordinatensystem der Maschine passen
- **Übersprungene Befehle prüfen** — Eine Benachrichtigung listet Befehle, die Rayforge nicht
  anwenden konnte

### Einschränkungen

- **Keine Lasereinstellungen** — Geschwindigkeit und Leistung werden nicht importiert
- **Verfahrwege ausgeschlossen** — Nur Schneidbewegungen werden zu Geometrie
- **Nicht unterstützte Befehle** — Bohrzyklen, Werkzeugradiuskorrektur, Unterprogramme und
  Nicht-XY-Ebenen werden übersprungen

---

## Illustrator- und EPS-Import

Illustrator-Dateien, die mit PDF-Kompatibilität gespeichert wurden (Standard seit Illustrator 9),
werden genau wie PDF-Dateien importiert, mit derselben Ebenenauswahl und denselben Tracing-Optionen.

EPS-, PostScript- und ältere Illustrator-Dateien werden zuerst mit Ghostscript in PDF umgewandelt.
Installiere Ghostscript, wenn der Importdialog meldet, dass es fehlt. Wenn du es nicht installieren
kannst, exportiere das Design im Ursprungsprogramm als PDF oder SVG.

---

## Rasterbild-Import (PNG, JPG, BMP, GIF, TIFF, WebP)

Rasterbilder werden **traced**, um Vektorpfade über den Import-Dialog zu erstellen.

### Tracing-Prozess im Dialog

**Wie es funktioniert:**

1. **Bild geladen** in den Import-Dialog
2. **Live-Vorschau** zeigt das Tracing-Ergebnis
3. **Tracing-Einstellungen** können in Echtzeit angepasst werden
4. **Vektorpfade erstellt** aus den getraceten Kanten
5. **Pfade hinzugefügt** zum Dokument als Werkstücke beim Importieren

### Tracing-Konfiguration im Dialog

Der Import-Dialog bietet diese einstellbaren Parameter:

| Parameter              | Beschreibung           | Effekt                                                     |
| ---------------------- | ---------------------- | ---------------------------------------------------------- |
| **Auto-Schwellenwert** | Automatische Erkennung | Wenn aktiviert, automatisch optimalen Schwellenwert finden |
| **Schwellenwert**      | Schwarz/Weiß-Grenze    | Niedriger = mehr Details, höher = einfacher                |
| **Invertieren**        | Farben umkehren        | Helle Objekte auf dunklem Hintergrund tracen               |

**Standardeinstellungen** funktionieren gut für die meisten Bilder. Der Dialog zeigt eine
Live-Vorschau, die sich beim Anpassen dieser Einstellungen aktualisiert, sodass du das Tracing vor
dem Importieren feinabstimmen kannst.

### Bilder für Tracing vorbereiten

**Für beste Ergebnisse:**

1. **Hoher Kontrast:**
   - Helligkeit/Kontrast im Bildeditor anpassen
   - Klare Unterscheidung zwischen Vordergrund und Hintergrund

2. **Sauberer Hintergrund:**
   - Rauschen und Artefakte entfernen
   - Solider weißer oder transparenter Hintergrund

3. **Angemessene Auflösung:**
   - 300-500 DPI für Fotos
   - Zu hoch = langsames Tracing, zu niedrig = schlechte Qualität

4. **Auf Inhalt zuschneiden:**
   - Unnötige Ränder entfernen
   - Auf den zu gravierenden/schneidenden Bereich fokussieren

5. **In Schwarz-Weiß konvertieren:**
   - Für Schneiden: reines S/W
   - Für Gravur: Graustufen sind in Ordnung

**Bildbearbeitungswerkzeuge:**

- GIMP (kostenlos)
- Photoshop
- Krita (kostenlos)
- Paint.NET (kostenlos, Windows)

### Tracing-Qualität

**Gute Tracing-Kandidaten:**

- Logos mit klaren Kanten
- Hochkontrastbilder
- Strichzeichnungen und Skizzen
- Text (obwohl besser als Vektor)

**Schlechte Tracing-Kandidaten:**

- Niedrigauflösende Bilder
- Fotos mit weichen Kanten
- Bilder mit Farbverläufen
- Sehr detaillierte oder komplexe Fotos

---

## Verwandte Seiten

- [Unterstützte Formate](formats) - Detaillierte Formatspezifikationen
- [G-Code exportieren](exporting) - Ausgabeoptionen
- [Schnellstart](../getting-started/quick-start.md) - Erstes Import-Tutorial
- [Farbregeln](../application-settings/color-rules.md) - SVG-Farben Schritttypen beim Import
  zuordnen
