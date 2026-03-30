# Addon-Manifest

Jedes Addon benötigt eine `rayforge-addon.yaml`-Datei in seinem Stammverzeichnis. Dieses Manifest informiert Rayforge über dein Addon – seinen Namen, was es bereitstellt und wie es geladen wird.

## Grundstruktur

Hier ist ein vollständiges Manifest mit allen gängigen Feldern:

```yaml
name: my_custom_addon
display_name: "My Custom Addon"
description: "Adds support for the XYZ laser cutter."
api_version: 9
url: https://github.com/username/my-custom-addon

author:
  name: Jane Doe
  email: jane@example.com

depends:
  - rayforge>=0.27.0

requires:
  - some-other-addon>=1.0.0

provides:
  backend: my_addon.backend
  frontend: my_addon.frontend
  assets:
    - path: assets/profiles.json
      type: profiles

license:
  name: MIT
```

## Pflichtfelder

### `name`

Eine eindeutige Kennung für dein Addon. Diese muss ein gültiger Python-Modulname sein – nur Buchstaben, Zahlen und Unterstriche, und sie darf nicht mit einer Zahl beginnen.

```yaml
name: my_custom_addon
```

### `display_name`

Ein lesbarer Name, der in der UI angezeigt wird. Dieser kann Leerzeichen und Sonderzeichen enthalten.

```yaml
display_name: "My Custom Addon"
```

### `description`

Eine kurze Beschreibung, was dein Addon tut. Diese erscheint im Addon-Manager.

```yaml
description: "Adds support for the XYZ laser cutter."
```

### `api_version`

Die API-Version, auf die dein Addon abzielt. Diese muss mindestens 1 (die minimal unterstützte Version) und höchstens die aktuelle Version (9) sein. Die Verwendung einer höheren als der unterstützten Version führt dazu, dass dein Addon die Validierung nicht besteht.

```yaml
api_version: 9
```

Siehe die [Hooks](./addon-hooks.md#api-versionshistorie)-Dokumentation für Änderungen in jeder Version.

### `author`

Informationen über den Addon-Autor. Das Feld `name` ist erforderlich; `email` ist optional, aber empfohlen, damit Benutzer dich kontaktieren können.

```yaml
author:
  name: Jane Doe
  email: jane@example.com
```

## Optionale Felder

### `url`

Eine URL zur Homepage oder zum Repository deines Addons.

```yaml
url: https://github.com/username/my-custom-addon
```

### `depends`

Versionseinschränkungen für Rayforge selbst. Gib die Mindestversion an, die dein Addon benötigt.

```yaml
depends:
  - rayforge>=0.27.0
```

### `requires`

Abhängigkeiten von anderen Addons. Liste Addon-Namen mit Versionseinschränkungen auf.

```yaml
requires:
  - some-other-addon>=1.0.0
```

### `version`

Die Versionsnummer deines Addons. Diese wird normalerweise automatisch aus Git-Tags ermittelt, kann aber explizit angegeben werden. Verwende semantische Versionierung (z. B. `1.0.0`).

```yaml
version: 1.0.0
```

## Einstiegspunkte

Der Abschnitt `provides` definiert, was dein Addon zu Rayforge beisteuert.

### Backend

Das Backend-Modul wird sowohl im Hauptprozess als auch in Worker-Prozessen geladen. Verwende dieses für Maschinentreiber, Schritttypen, Ops-Produzenten und jede Kernfunktionalität.

```yaml
provides:
  backend: my_addon.backend
```

Der Wert ist ein gepunkteter Python-Modulpfad relativ zu deinem Addon-Verzeichnis.

### Frontend

Das Frontend-Modul wird nur im Hauptprozess geladen. Verwende dieses für UI-Komponenten, GTK-Widgets und alles, das das Hauptfenster benötigt.

```yaml
provides:
  frontend: my_addon.frontend
```

### Assets

Du kannst Asset-Dateien bündeln, die Rayforge erkennen wird. Jedes Asset hat einen Pfad und einen Typ:

```yaml
provides:
  assets:
    - path: assets/profiles.json
      type: profiles
    - path: assets/templates
      type: templates
```

Der `path` ist relativ zu deinem Addon-Stamm und muss existieren. Asset-Typen werden von Rayforge definiert und können Dinge wie Maschinenprofile, Materialbibliotheken oder Vorlagen umfassen.

## Lizenzinformationen

Das Feld `license` beschreibt, wie dein Addon lizenziert ist. Für kostenlose Addons gib einfach den Lizenznamen mit einem SPDX-Bezeichner an:

```yaml
license:
  name: MIT
```

Gängige SPDX-Bezeichner sind `MIT`, `Apache-2.0`, `GPL-3.0` und `BSD-3-Clause`.

## Kostenpflichtige Addons

Rayforge unterstützt kostenpflichtige Addons durch Gumroad-Lizenzvalidierung. Wenn du dein Addon verkaufen möchtest, kannst du es so konfigurieren, dass es eine gültige Lizenz erfordert, bevor es funktioniert.

### Grundlegende kostenpflichtige Konfiguration

```yaml
license:
  name: BSL-1.1
  required: true
  purchase_url: https://gum.co/my-addon
```

Wenn `required` auf true gesetzt ist, prüft Rayforge auf eine gültige Lizenz, bevor dein Addon geladen wird. Die `purchase_url` wird Benutzern angezeigt, die keine Lizenz haben.

### Gumroad-Produkt-ID

Füge deine Gumroad-Produkt-ID hinzu, um die Lizenzvalidierung zu aktivieren:

```yaml
license:
  name: BSL-1.1
  required: true
  purchase_url: https://gum.co/my-addon
  product_id: "abc123def456"
```

Für mehrere Produkt-IDs (z. B. verschiedene Preistufen):

```yaml
license:
  name: BSL-1.1
  required: true
  purchase_url: https://gum.co/my-addon
  product_ids:
    - "abc123def456"
    - "xyz789ghi012"
```

### Vollständiges Beispiel für ein kostenpflichtiges Addon

Hier ist ein vollständiges Manifest für ein kostenpflichtiges Addon:

```yaml
name: premium_laser_pack
display_name: "Premium Laser Pack"
description: "Advanced features for professional laser cutting."
api_version: 9
url: https://example.com/premium-laser-pack

author:
  name: Your Name
  email: you@example.com

depends:
  - rayforge>=0.27.0

provides:
  backend: premium_pack.backend
  frontend: premium_pack.frontend

license:
  name: BSL-1.1
  required: true
  purchase_url: https://gum.co/premium-laser-pack
  product_ids:
    - "standard_tier_id"
    - "pro_tier_id"
```

### Lizenzstatus im Code prüfen

In deinem Addon-Code kannst du prüfen, ob eine Lizenz gültig ist:

```python
@hookimpl
def rayforge_init(context):
    if context.license_validator:
        # Check if user has a valid license for your product
        is_valid = context.license_validator.is_product_valid("your_product_id")
        if not is_valid:
            # Optionally show a message or limit functionality
            logger.warning("License not found - some features disabled")
```

## Validierungsregeln

Rayforge validiert dein Manifest beim Laden des Addons. Hier sind die Regeln:

Der `name` muss ein gültiger Python-Bezeichner sein (Buchstaben, Zahlen, Unterstriche, keine führenden Zahlen). Die `api_version` muss eine ganze Zahl zwischen 1 und der aktuellen Version sein. Das Feld `author.name` darf nicht leer sein oder Platzhaltertext wie "your-github-username" enthalten. Einstiegspunkte müssen gültige Modulpfade sein und die Module müssen existieren. Asset-Pfade müssen relativ sein (kein `..` oder führendes `/`) und die Dateien müssen existieren.

Wenn die Validierung fehlschlägt, protokolliert Rayforge einen Fehler und überspringt dein Addon. Prüfe die Konsolenausgabe während der Entwicklung, um diese Probleme zu erkennen.
