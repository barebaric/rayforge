# Bildbearbeitung

Das Bildbearbeitungs-Addon verändert, wie das Bild eines Werkstücks aussieht, ohne die importierte
Datei zu ändern. Jede Bearbeitung wird mit dem Werkstück gespeichert, sodass du sie rückgängig
machen, später entfernen oder dieselbe Datei mehrmals mit unterschiedlichen Bearbeitungen verwenden
kannst.

Die Leinwand, die 3D-Vorschau und jede Operation verwenden das bearbeitete Bild. Was du auf der
Leinwand siehst, ist das, was der Laser graviert.

## Bild invertieren

Wähle ein oder mehrere Werkstücke aus und wähle **Tools - Invert Image**. Helle Bereiche werden
dunkel und dunkle Bereiche werden hell; transparente Bereiche bleiben transparent, sodass der
Hintergrund weiterhin nicht graviert wird.

![Ein Graustufenfoto von Bergen und einem See, wie importiert](/images/image-edit-invert-before.webp)

![Dasselbe Foto nach Invert Image](/images/image-edit-invert-after.webp)

Wähle es erneut, um die Invertierung zu entfernen. Bei einer gemischten Auswahl werden die
Werkstücke, die noch nicht invertiert sind, invertiert, sodass die gesamte Auswahl am Ende gleich
ist.

Verwende es für Lithophanen und Prägen in einer Gravur-Operation, bei der helle Bereiche tief
graviert werden sollen, oder um helle Motive auf dunklem Hintergrund zu gravieren.

## Bild anpassen

Wähle ein oder mehrere Werkstücke aus und wähle **Tools - Adjust Image...**, um ein Foto vor dem
Gravieren oder Nachzeichnen zu korrigieren:

- **Brightness** (Helligkeit) hellt das Bild auf oder dunkelt es ab.
- **Contrast** (Kontrast) zieht die Tonwerte vom mittleren Grau weg oder zu ihm hin.
- **Gamma** über 1 hellt die Mitteltöne auf, unter 1 dunkelt es sie ab; Schwarz und Weiß bleiben,
  wie sie sind.
- **Sharpen** (Schärfen) verstärkt Kanten und feine Details (eine Unscharfmaskierung). **Sharpen
  Radius** legt die Größe der Details in Millimetern fest, sodass das Ergebnis bei jedem
  Zeilenabstand und jeder Zoomstufe gleich ist.

Die Leinwand zeigt die Änderungen schon während der Bearbeitung. **Apply** übernimmt sie als einen
einzigen Rückgängig-Schritt; wenn du den Dialog ohne Übernehmen schließt, werden sie verworfen, und
**Reset** setzt alle Werte auf neutral zurück. Transparente Bereiche bleiben transparent.

Der Dialog öffnet sich mit den Werten des ersten ausgewählten Werkstücks, und Apply gibt jedem
ausgewählten Werkstück dieselben Werte. Das Schärfen erfolgt vor den Tonwertkorrekturen, und andere
Bearbeitungen wie Invert Image behalten ihren Platz.

## Ältere Projekte

Vor diesem Addon war Invertieren eine Einstellung der Gravur-Operation. Wenn du ein älteres Projekt
öffnest, das sie verwendet hat, verschiebt Rayforge die Einstellung auf die Werkstücke dieser Ebene,
sodass das Ergebnis gleich bleibt.

## Wenn das Addon deaktiviert ist

Ein Projekt behält seine Bildbearbeitungen, wenn das Addon deaktiviert oder nicht installiert ist.
Rayforge teilt dir mit, dass das Projekt Funktionen verwendet, die nicht verfügbar sind, und zeigt
die Bilder ohne diese Bearbeitungen an, bis das Addon wieder aktiviert ist.
