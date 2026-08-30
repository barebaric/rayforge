# Materialien

![Material-Einstellungen](/screenshots/app-settings-materials.webp)

Material-Bibliotheken in Rayforge ermöglichen es dir, Material-Sammlungen für deine Laserschneide-
und Gravurprojekte zu organisieren und zu verwalten. Diese Anleitung erklärt den Unterschied
zwischen Kern- und Benutzer-Bibliotheken und wie du deine eigenen Bibliotheken erstellen und
Materialien hinzufügen kannst.

<!-- prettier-ignore-start -->
:::note
Das Zuweisen eines Materials zu einem Materialstück beeinflusst sowohl sein optisches
Erscheinungsbild in der 2D- und 3D-Ansicht als auch, welche [Rezepte](recipes.md) darauf angewendet
werden: materialspezifische Rezepte werden anhand des zugewiesenen Materials abgeglichen. In
zukünftigen Versionen werden Materialien verwendet, um weitere funktionale Parameter abzuleiten.
:::
<!-- prettier-ignore-end -->

## Eine neue Bibliothek erstellen

Um deine eigene Materialbibliothek zu erstellen:

1. Öffne das Menü **Einstellungen** und wähle **Materialien**
2. Klicke auf die Schaltfläche **Neue Bibliothek hinzufügen**, um eine neue Bibliothek zu erstellen
3. Gib einen beschreibenden Namen für deine Bibliothek ein (z.B. "Meine Werkstatt-Materialien")
4. Klicke auf **Erstellen** zum Fertigstellen

Deine neue Bibliothek wird im Benutzerdaten-Verzeichnis erstellt und ist sofort verfügbar.

## Materialien zu Bibliotheken hinzufügen

### Ein neues Material erstellen

1. Wähle die Bibliothek aus, zu der du das Material hinzufügen möchtest
2. Klicke auf die Schaltfläche **Neues Material hinzufügen** in der Materialliste
3. Fülle die Materialeigenschaften aus:
   - **Name**: Lesbarer Name
   - **Kategorie**: Gruppierungskategorie (z.B. "Holz", "Acryl")
   - **Aussehen**: Visuelle Eigenschaften (siehe unten)
4. Klicke auf **Speichern**, um das Material zur Bibliothek hinzuzufügen

### Materialeigenschaften erklärt

#### Name

- Lesbarer Name, der in der Schnittstelle angezeigt wird
- Kann Leerzeichen und Sonderzeichen enthalten

#### Kategorie

- Wird zum Organisieren von Materialien innerhalb der Bibliothek verwendet
- Häufige Kategorien: Holz, Acryl, Metall, Papier, Leder
- Du kannst benutzerdefinierte Kategorien nach Bedarf erstellen

#### Textur

Ein Texturbild (WebP oder PNG), das über die Materialoberfläche gekachelt wird. Wenn eine Textur
festgelegt ist, wird das Material mit der Textur statt mit einer einfarbigen Fläche dargestellt.
Texturen können mit dem Skript `scripts/optimize_material_textures.py` in WebP optimiert werden, um
die Materialdateien klein zu halten.

#### Texturskalierung

Die Größe (in mm), die eine Texturkachel auf dem Material abdeckt. Kleinere Werte wiederholen die
Textur häufiger auf derselben Fläche.

#### Farbe

Eine optionale Tönungsfarbe. Wenn sie festgelegt ist, wird die Textur des Materials mit dieser Farbe
eingefärbt; andernfalls wird die Textur unverändert angezeigt. Dadurch kann ein einzelnes
texturiertes Material (z.B. "Acryl") mehrere Farbvarianten abdecken: Die Farbe wird pro
Materialstück im Dialog [Materialeigenschaften](../features/stock-handling.md) angewendet. Die Farbe
wird nur für das visuelle Erscheinungsbild auf der Arbeitsfläche verwendet - sie beeinflusst den
Laserpfad in keiner Weise.

#### Rauheit

Ein Wert von 0-1, der beschreibt, wie rau oder poliert die Oberfläche in der 3D-Ansicht erscheint.
Niedrigere Werte wirken glänzend, höhere Werte matt.

#### Metallisch

Ein Wert von 0-1, der beschreibt, ob die Oberfläche in der 3D-Ansicht Licht wie ein Metall
reflektiert. Setze 1 für metallische Materialien, 0 für nicht-metallische.

#### Absorption {#absorption}

<!-- prettier-ignore-start -->
:::note[Neu in 1.11]
Absorptionsdaten steuern das
[physikalische Brennmodell](../ui/3d-preview.md#physical-burn-model) in der 3D-Vorschau.
:::
<!-- prettier-ignore-end -->

Wellenlängenabhängige Absorptionskoeffizienten (0–1) beschreiben, wie viel der Laserenergie ein
Material bei einer bestimmten Wellenlänge absorbiert. Die 3D-Vorschau verwendet diese zusammen mit
der Wellenlänge, optischen Wattzahl und Punktgröße deines Laserkopfs, um die Fluenz (J/cm²) zu
berechnen und einen physikalisch motivierten Verbrandungseffekt auf dem Material zu rendern.

Füge einen `absorption`-Block unter `appearance` in der YAML-Datei des Materials hinzu:

```yaml
appearance:
  absorption:
    blue: 0.7 # ~445 nm Diodenlaser
    ir: 0.25 # ~1064 nm Faser-/IR-Laser
    co2: 0.9 # ~10600 nm CO2-Laser
  # ...weitere Aussehens-Eigenschaften
```

| Band   | Repräsentative Wellenlänge | Typische Laser    |
| ------ | -------------------------- | ----------------- |
| `blue` | 445 nm                     | Blaue Diodenlaser |
| `ir`   | 1064 nm                    | Faserlaser        |
| `co2`  | 10600 nm                   | CO2-Röhrenlaser   |

Wenn ein Band fehlt, wird ein konservativer Standardwert verwendet. Die mitgelieferte
Materialbibliothek enthält recherchierte Absorptionswerte für alle enthaltenen Materialien; das
Brennmodell ist noch nicht vollständig kalibriert, daher sind Beiträge mit Daten aus der Praxis
willkommen.

## Bestehende Materialien verwalten

### Materialien bearbeiten

1. Wähle das Material aus, das du bearbeiten möchtest
2. Klicke auf die Schaltfläche **Bearbeiten**
3. Ändere die gewünschten Eigenschaften
4. Klicke auf **Speichern**, um die Änderungen anzuwenden

### Materialien löschen

1. Wähle das Material aus, das du löschen möchtest
2. Klicke auf die Schaltfläche **Löschen**
3. Bestätige das Löschen im Dialog

<!-- prettier-ignore-start -->
:::warning
Das Löschen eines Materials ist dauerhaft und kann nicht rückgängig gemacht werden.
:::
<!-- prettier-ignore-end -->
