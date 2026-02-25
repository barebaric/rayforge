# Nutzungsanalyse

Rayforge bietet eine optionale anonyme Nutzungsanalyse, um zu verstehen, wie die Anwendung verwendet wird und die zukünftige Entwicklung zu priorisieren. Diese Seite erklärt, was wir erfassen, wie es funktioniert und Ihre Privatsphäre.

## Rein freiwillig

Die Nutzungsanalyse ist **komplett optional**. Beim ersten Start von Rayforge werden Sie gefragt, ob Sie teilnehmen möchten:

- **Ja**: Anonyme Nutzungsdaten werden an unseren Analyseserver gesendet
- **Nein**: Es werden niemals Daten erfasst oder übertragen

Sie können diese Entscheidung jederzeit in den allgemeinen Einstellungen ändern.

## Was wir erfassen

Wenn aktiviert, erfassen wir nur anonyme Seitenaufrufdaten – ähnlich wie Website-Analysen. Das können wir sehen:

| Daten                    | Beispiel                  |
| ------------------------ | ------------------------- |
| Bildschirmauflösung      | 1920x1080                 |
| Spracheinstellung        | de-DE                     |
| Seiten/Dialoge angesehen | /machine-settings/general |
| Zeit auf Seite           | 6m 3s                     |

## Was wir sehen

Hier ist ein Beispiel, wie das Analyse-Dashboard aussieht:

| Pfad                      | Besucher | Besuche | Aufrufe | Absprungrate | Besuchsdauer |
| ------------------------- | -------- | ------- | ------- | ------------ | ------------ |
| /                         | 1        | 1       | 5       | 0%           | 27m 35s      |
| /machine-settings/general | 1        | 1       | 5       | 0%           | 27m 27s      |
| /view/3d                  | 1        | 1       | 2       | 0%           | 25m 14s      |
| /camera-alignment-dialog  | 1        | 1       | 2       | 0%           | 6m 3s        |
| /machine-settings/camera  | 1        | 1       | 2       | 0%           | 6m 16s       |
| /settings/general         | 1        | 1       | 2       | 0%           | 16m 36s      |
| /step-settings/rasterizer | 1        | 1       | 2       | 0%           | 11s          |

## Was wir NICHT erfassen

Wir verpflichten uns zu Ihrer Privatsphäre:

- **Keine persönlichen Informationen** – Keine Namen, E-Mails oder Konten
- **Keine Dateiinhalte** – Ihre Designs und Projekte bleiben privat
- **Keine Maschinenkennungen** – Keine Seriennummern oder eindeutigen IDs
- **Keine IP-Adressen gespeichert** – Wir nutzen Umami Analytics, das IPs nicht speichert
- **Keine seitenübergreifende Verfolgung** – Daten sind nur auf Rayforge beschränkt

## Warum wir analysieren

Nutzungsdaten helfen uns:

- **Beliebte Funktionen identifizieren** – Wissen, was gut funktioniert
- **Schmerzpunkte finden** – Sehen, wo Nutzer Zeit verbringen oder stecken bleiben
- **Entwicklung priorisieren** – Uns auf Funktionen konzentrieren, die Menschen tatsächlich nutzen
- **Vielfalt verstehen** – Wissen, welche Sprachen und Bildschirmgrößen wir unterstützen sollten

## Wie es funktioniert

Rayforge nutzt [Umami](https://umami.is/), eine Open-Source, datenschutzfreundliche Analyseplattform. Die Analyse:

- Sendet kleine HTTP-Anfragen im Hintergrund
- Beeinflusst nicht die Anwendungsleistung
- Funktioniert offline (fehlgeschlagene Anfragen werden still ignoriert)
- Verwendet einen generischen User-Agent zur Verhinderung von Fingerprinting

## Analyse deaktivieren

Sie können die Analyse jederzeit deaktivieren:

1. Öffnen Sie **Einstellungen** → **Allgemein**
2. Schalten Sie **Anonyme Nutzungsstatistiken senden** aus

Wenn deaktiviert, werden absolut keine Daten gesendet.

## Verwandte Seiten

- **[Anwendungseinstellungen](../ui/settings)** – Analyse-Einstellungen konfigurieren
