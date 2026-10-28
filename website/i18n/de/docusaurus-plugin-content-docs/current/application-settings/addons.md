# Addons

![Addon-Einstellungen](/screenshots/app-settings-addons.webp)

Die Addons-Seite zeigt installierte Erweiterungsaddons und ermöglicht es dir, diese zu verwalten.

## Übersicht

Addons sind Erweiterungen, die Rayforge Funktionalität hinzufügen. Sie können bereitstellen:

- Neue Operationen
- Import/Export-Filter
- Maschinen-Treiber
- UI-Erweiterungen
- Material-Bibliotheken

## Installierte Addons

Die Addonliste zeigt alle installierten Addons mit:

- **Name**: Der Anzeigename des Addons
- **Version**: Aktuell installierte Version
- **Beschreibung**: Kurze Beschreibung der Funktionalität
- **Status**: Ob aktiviert oder deaktiviert

## Addons verwalten

### Addons installieren

1. Klicke auf die Schaltfläche **Installieren**
2. Wähle eine Addondatei (`.zip` oder Addon-Repository)
3. Bestätige die Installation (bei Community-Addons muss eine Vertrauenswarnung bestätigt werden)
4. Warte, bis die Installation abgeschlossen ist

### Addons aktualisieren

Wenn Updates verfügbar sind, erscheint ein Indikator im Addon-Registry-Dialog. Klicke auf
**Aktualisieren**, um verfügbare Updates zu installieren.

### Addons entfernen

1. Wähle das zu entfernende Addon aus
2. Klicke auf die Schaltfläche **Entfernen**
3. Bestätige das Entfernen

<!-- prettier-ignore-start -->
:::warning
Das Entfernen eines Addons kann Dokumente betreffen, die dessen Funktionen verwenden.
:::
<!-- prettier-ignore-end -->

## Community-Addons

Addons, die nicht vom Rayforge-Team erstellt wurden, werden in den Addonlisten mit einem eigenen
Symbol als Community-Addons gekennzeichnet.

Beim Installieren eines Community-Addons zeigt ein Bestätigungsdialog an, dass das Addon nicht von
Rayforge selbst bereitgestellt wird, und enthält einen Link zum Repository des Addons. Öffne den
Link im Browser, um den Code und dessen Autor zu prüfen, bevor du fortfährst.

<!-- prettier-ignore-start -->
:::warning
Addons haben dieselben Zugriffsrechte auf deinen Computer und deine Dateien wie Rayforge selbst.
Installiere Community-Addons nur, wenn du ihren Autoren vertraust.
:::
<!-- prettier-ignore-end -->

## Addonquellen

Addons können installiert werden aus:

- **Repositories**: Git-Repositories, die Addons enthalten
- **Offizielles Register**: Kuratierte Addons vom Rayforge-Team

## Verwandte Themen

- [Maschinen](machines) - Maschinenkonfiguration
- [Materialien](materials) - Material-Bibliotheken
- [Rezepte](recipes) - Operations-Rezepte
