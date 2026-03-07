# KI-Anbieter

![KI-Anbieter Einstellungen](/screenshots/application-ai.png)

Konfiguriere KI-Anbieter, die Addons nutzen können, um Rayforge um
intelligente Funktionen zu erweitern.

## Funktionsweise

Addons können konfigurierte KI-Anbieter nutzen, ohne eigene API-Schlüssel
zu benötigen. Dies zentralisiert deine KI-Konfiguration und ermöglicht
dir die Kontrolle darüber, welche Anbieter für Addons verfügbar sind.

## Anbieter hinzufügen

1. Klicke auf **Anbieter hinzufügen**, um eine neue Anbieterkonfiguration
   zu erstellen
2. Gib einen **Namen** ein, um diesen Anbieter zu identifizieren
3. Lege die **Basis-URL** auf den API-Endpunkt deines KI-Dienstes fest
4. Gib deinen **API-Schlüssel** zur Authentifizierung ein
5. Gib ein **Standardmodell** an, das mit diesem Anbieter verwendet
   werden soll
6. Klicke auf **Testen**, um zu überprüfen, ob deine Konfiguration
   funktioniert

## Anbietertypen

### OpenAI-kompatibel

Dieser Anbietertyp funktioniert mit jedem Dienst, der das OpenAI-API-Format
verwendet. Dazu gehören verschiedene Cloud-Anbieter und selbst gehostete
Lösungen.

Die Standard-Basis-URL ist auf die API von OpenAI eingestellt, aber du
kannst sie auf jeden kompatiblen Dienst ändern.

## Anbieter verwalten

- **Aktivieren/Deaktivieren**: Schalte einen Anbieter ein oder aus,
  ohne ihn zu löschen
- **Als Standard festlegen**: Klicke auf das Häkchen-Symbol, um einen
  Anbieter zum Standard zu machen
- **Löschen**: Entferne einen Anbieter, den du nicht mehr benötigst

:::warning
Deine API-Schlüssel werden lokal auf deinem Computer gespeichert und
niemals an Dritte weitergegeben.
:::

## Verwandte Themen

- [Addons](addons) - Addons installieren und verwalten
- [Maschinen](machines) - Maschinenkonfiguration
- [Materialien](materials) - Materialbibliotheken
