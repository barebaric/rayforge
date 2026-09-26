# Lasereinstellungen

Die Laser-Seite in den Maschineneinstellungen konfiguriert deine Laserköpfe und deren Eigenschaften.

![Lasereinstellungen](/screenshots/machine-settings-laser.webp)

## Laserköpfe

Rayforge unterstützt Maschinen mit mehreren Laserköpfen. Jeder Laserkopf hat seine eigene
Konfiguration.

### Einen Laserkopf hinzufügen

Klicke auf die Schaltfläche **Laser hinzufügen**, um eine neue Laserkopf-Konfiguration zu erstellen.

### Laserkopf-Eigenschaften

Jeder Laserkopf hat die folgenden Einstellungen:

#### Name

Ein beschreibender Name für diesen Laserkopf.

Beispiele:

- "10W Diode"
- "CO2-Röhre"
- "Infrarotlaser"

#### Werkzeugnummer

Der Werkzeugindex für diesen Laserkopf. Im G-Code mit dem T-Befehl verwendet.

- Ein-Kopf-Maschinen: 0 verwenden
- Multi-Kopf-Maschinen: Eindeutige Nummern zuweisen (0, 1, 2, usw.)

#### Maximale Leistung

Der maximale Leistungswert für deinen Laser.

- **GRBL typisch**: 1000 (S0-S1000 Bereich)
- **Einige Controller**: 255 (S0-S255 Bereich)
- **Prozentmodus**: 100 (S0-S100 Bereich)

Dieser Wert sollte mit der $30-Einstellung deiner Firmware übereinstimmen.

#### Rahmen-Leistung

Der Leistungswert, der für Rahmen-Operationen verwendet wird (Umreißen ohne Schneiden).

- Auf 0 setzen, um Rahmen zu deaktivieren
- Passe ihn basierend auf deinem Laser und Material an

#### Rahmen-Geschwindigkeit

Die Geschwindigkeit, mit der sich der Laserkopf während des Einrahmens bewegt. Dies wird pro
Laserkopf eingestellt, sodass du bei Maschinen mit mehreren Lasern mit unterschiedlichen
Eigenschaften eine angemessene Geschwindigkeit für jeden wählen kannst. Langsamere Geschwindigkeiten
machen den Rahmen-Pfad leichter visuell verfolgbar.

#### Fokus-Leistung

Die Leistungsstufe, die verwendet wird, wenn der Fokusmodus aktiviert ist. Der Fokusmodus schaltet
den Laser mit niedriger Leistung ein, um als "Laserzeiger" zur Positionierung zu dienen.

- Auf 0 setzen, um die Fokusmodus-Funktion zu deaktivieren
- Verwende für visuelle Ausrichtung und Positionierung

<!-- prettier-ignore-start -->
:::tip[Fokusmodus verwenden]
Klicke auf die Fokus-Taste (Laser-Symbol) in der Symbolleiste, um den
Fokusmodus umzuschalten. Der Laser wird bei dieser Leistungsstufe eingeschaltet und hilft dir, genau
zu sehen, wo der Laser positioniert ist. Siehe
[Werkstückpositionierung](../features/workpiece-positioning.md) für weitere Informationen.
:::
<!-- prettier-ignore-end -->

#### Punktgröße

Die physische Größe deines fokussierten Laserstrahls in Millimetern.

- Gib sowohl X- als auch Y-Abmessungen ein
- Die meisten Laser haben einen runden Punkt (z.B. 0.1 x 0.1)
- Beeinflusst Gravurqualitäts-Berechnungen

<!-- prettier-ignore-start -->
:::tip[Punktgröße messen]
Um deine Punktgröße zu messen:

1. Feuere einen kurzen Impuls bei niedriger Leistung auf ein Testmaterial
2. Miss die resultierende Markierung mit einer Schieblehre
3. Verwende den Durchschnitt mehrerer Messungen
:::
<!-- prettier-ignore-end -->

#### Farbe

Die Farbe, die zum Anzeigen der Operationen dieses Lasers (Schnitte und Gravuren) im Canvas und in
der 3D-Vorschau verwendet wird. Dies hilft dir, visuell zu unterscheiden, welcher Laser welche
Operation durchführen wird, wenn du mit mehreren Laserköpfen arbeitest.

- Klicke auf die Farbauswahl, um einen Farbwähler zu öffnen
- Wähle eine Farbe, die gut mit deiner Materialvorschau kontrastiert
- Standardfarben werden automatisch zugewiesen

<!-- prettier-ignore-start -->
:::tip[Multi-Laser-Workflows]
Bei der Verwendung mehrerer Laserköpfe erleichtert das Zuweisen
unterschiedlicher Farben zu jedem Laser es, zu erkennen, welche Operationen von welchem Laser
durchgeführt werden. Verwende beispielsweise Rot für deinen Hauptschneidelaser und Blau für einen
sekundären Gravurlaser.
:::
<!-- prettier-ignore-end -->

#### Lasertyp

Wähle den Typ des Laserkopfs aus dem Dropdown-Menü:

- **Diode**: Standard-Diodenlaser (am häufigsten bei Hobby-Maschinen)
- **CO2**: CO2-Röhrenlaser
- **Faser**: Faserlaser

Wenn CO2 oder Faser ausgewählt ist, werden zusätzliche **PWM-Einstellungen** sichtbar (siehe unten).
Für Diodenlaser ist der PWM-Bereich ausgeblendet, da er nicht zutreffend ist.

Der Lasertyp legt auch eine Standard-**Wellenlänge** (verwendet vom physikalischen Brennmodell)
fest, wenn kein expliziter Wert unten eingegeben wird.

#### Wellenlänge (nm)

Die Emissionswellenlänge deines Lasers in Nanometern. Diese speist das
[physikalische Brennmodell](../ui/3d-preview.md#physical-burn-model) in der 3D-Vorschau: Zusammen
mit den [Absorptions-](../application-settings/materials.md#absorption) daten des Materials bestimmt
sie, wie viel Laserenergie das Material absorbiert.

Wenn auf 0 gesetzt, greift Rayforge auf die typische Wellenlänge für den gewählten Lasertyp zurück
(z.B. 445 nm für Diode, 1064 nm für Faser, 10600 nm für CO2).

#### Max. optische Leistung (W)

Die optische Ausgangsleistung deines Lasers bei voller Leistung in Watt. Dies ist die tatsächliche
Lichtausgabe, nicht der elektrische Eingang. Zusammen mit der Punktgröße und Scangeschwindigkeit
bestimmt sie die Fluenz (J/cm²), die vom
[physikalischen Brennmodell](../ui/3d-preview.md#physical-burn-model) verwendet wird.

Wenn auf 0 gesetzt, wird ein mittlerer Desktop-Standardwert verwendet.

#### PWM-Einstellungen

Wenn ein CO2- oder Faserlaser-Typ ausgewählt ist, erscheinen folgende PWM-Steuerelemente:

- **PWM-Frequenz**: Die Standard-PWM-Frequenz in Hz für diesen Laserkopf. Typische Werte reichen von
  500 Hz bis mehrere kHz, je nach Controller und Netzteil.
- **Max. PWM-Frequenz**: Die Obergrenze für die Frequenzeinstellung. Dies verhindert die Eingabe von
  Werten, die deine Hardware nicht verarbeiten kann.
- **Pulsbreite**: Die Standard-Pulsbreite in Mikrosekunden. Steuert, wie lange jeder Puls während
  eines Zyklus eingeschaltet ist.
- **Min/Max Pulsbreite**: Grenzen für die Pulsbreiteneinstellung.

Diese Standardwerte werden an deine Operationsschritte übergeben, wo sie bei Bedarf pro Schritt
überschrieben werden können.

#### Zeiger-Offset

Wenn deine Maschine einen separaten Zeigerlaser (einen kleinen Laser mit rotem Punkt) in festem
Abstand zum Schneidstrahl hat, kannst du Rayforge diesen Abstand mitteilen, damit er ihn
kompensiert.

- **Zeiger-Offset verwenden**: Aktiviert die Kompensation. Standardmäßig aus.
- **Zeiger-Offset X / Y**: Der Abstand in Millimetern von der Position des Schneidstrahls zum
  Zeigerpunkt entlang der X- bzw. Y-Achse der Maschine.

Bei aktiviertem Offset ändert sich dreierlei:

1. **Arbeitsnullpunkt an aktueller Position setzen** (und die Tasten Nullen X / Nullen Y) legt den
   Arbeitsursprung dorthin, wo der _Zeigerpunkt_ das Werkstück markiert — nicht dorthin, wo der
   (unsichtbare) Schneidstrahl ist.
2. Die Leinwand zeigt einen gelben Zeigerpunkt neben dem roten Strahlpunkt, der markiert, wo sich
   der Zeigerpunkt auf deinem Material befindet.
3. Ein **Zeigerausrichtung**-Schalter wird im Verschieben-Popover verfügbar (siehe unten).

#### Zeigerausrichtung

Die Zeigerausrichtung ist ein Laufzeit-Schalter im Verschieben-Popover (das Kompass-Symbol neben der
Positionsanzeige). Ist er aktiv, werden alle absoluten Positionierungsoperationen — Verschieben
nach, die Ecken-Kürzel, das Anfahren des WKS-Ursprungs, Klicken zum Verschieben, Kopf hierher
bewegen und Einrahmen — verschoben, sodass der _Zeigerpunkt_ auf der anvisierten Position landet.
Auf der Leinwand ist stets genau ein Punkt gefüllt: Der Zeigerpunkt wird gefüllt gezeichnet, während
die Ausrichtung aktiv ist (der Strahlpunkt ist dann ein hollower Ring), und hohl, während sie aus
ist (der Strahlpunkt ist gefüllt).

Der typische Arbeitsablauf:

1. Verfahre die Maschine, bis der Zeigerpunkt deinen Referenzpunkt auf dem Werkstück markiert.
2. **Setze dort den Arbeitsnullpunkt** — bei aktiviertem Zeiger-Offset landet der Ursprung genau
   dort, wo der Zeiger gezeigt hat.
3. Aktiviere **Zeigerausrichtung** im Verschieben-Popover.
4. Rahmen und verschiebe mit dem Zeigerpunkt: Alles, was du anvisierst, wird vom Zeiger markiert.
5. Wenn du auf **Senden** drückst, erscheint eine Warnung: Ein normaler Job brennt mit dem Strahl an
   den WKS-Positionen. Du kannst einen **Probelauf mit Zeiger** ausführen (der Job läuft mit
   angewendetem Zeiger-Offset, sodass der Zeigerpunkt den Werkzeugpfad nachzeichnet, während der
   Strahl verschoben läuft — er brennt weiterhin mit Job-Leistung), die Ausrichtung ausschalten und
   brennen oder abbrechen.

Zwei Dinge werden bei einem normalen Senden nie verschoben: **Jog** (eine relative Bewegung braucht
keine Kompensation) und **Jobs** — Schneiden passiert immer mit dem Strahl an den WKS-Positionen,
daher ist deine G-Code-Ausgabe identisch, ob die Ausrichtung ein- oder ausgeschaltet ist. Nur der
ausdrückliche Probelauf mit Zeiger wendet den Offset auf einen Job an. Zusammen mit dem Nullen per
Zeigerpunkt bleibt alles konsistent: Der Ursprung liegt bei `Strahl + Offset`, das Anvisieren
verschiebt jedes Ziel um `-Offset`, und der Brennvorgang bleibt unverschoben.

Die Zeigerausrichtung ist eine sitzungsbezogene Einstellung: Sie wird nicht im Maschinenprofil
gespeichert und zurückgesetzt, wenn du die Maschine wechselst.

<!-- prettier-ignore-start -->
:::tip[Offset messen]
1. Verfahre die Maschine, bis der Zeigerpunkt eine sichtbare Stelle auf dem
   Werkstück markiert.
2. Aktiviere den Fokusmodus und verfahre, bis der *Schneidstrahl* an genau
   derselben Stelle eine Markierung brennt (oder bewege den Strahl vorsichtig
   bei Fokusleistung dorthin).
3. Der Offset ist die Zeigerposition minus der Strahlposition. Wenn der Zeiger
   zum Beispiel X=100 markierte und du den Strahl zu X=88 verfahren musstest,
   um dieselbe Stelle zu treffen, gib X = 12.0 ein — der Zeigerpunkt sitzt
   12 mm vor dem Strahl.

Wenn ein Testschnitt versetzt herauskommt, drehe das Vorzeichen der
betreffenden Achse um.
:::
<!-- prettier-ignore-end -->

<!-- prettier-ignore-start -->
:::note[Rotationsmodus]
Wenn der Rotationsaufsatz aktiv ist, wird die Y-Achse durch die
Rotationswalze ersetzt, sodass die Y-Komponente des Zeiger-Offsets nicht
sinnvoll anwendbar ist. Setze sie bei Rotationsjobs auf 0.
:::
<!-- prettier-ignore-end -->

#### 3D-Modell

Jedem Laserkopf kann ein 3D-Modell zugewiesen werden. Dieses Modell wird in der
[3D-Ansicht](../ui/3d-preview.md) gerendert und folgt dem Werkzeugweg während der Simulation.

Klicke auf die Modellauswahlzeile, um verfügbare Modelle zu durchsuchen. Sobald ein Modell
ausgewählt ist, kannst du dessen Skalierung, Rotation (X/Y/Z) und Fokusabstand an deinen physischen
Laserkopf anpassen.

## Siehe auch

- [Geräteeinstellungen](device) - GRBL Lasermodus-Einstellungen
- [Werkstückpositionierung](../features/workpiece-positioning.md) - Verwendung von Fokusmodus und
  anderen Positionierungsmethoden
