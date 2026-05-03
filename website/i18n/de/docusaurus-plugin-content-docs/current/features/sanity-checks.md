---
description: "Vor dem Ausführen oder Exportieren eines Auftrags prüft Rayforge automatisch auf häufige Probleme wie Grenzverletzungen, Arbeitsbereichverletzungen und No-Go-Zonen-Kollisionen."
---

# Auftrags-Plausibilitätsprüfungen

Vor dem Ausführen oder Exportieren eines Auftrags führt Rayforge automatisch
eine Reihe von Plausibilitätsprüfungen durch und zeigt die Ergebnisse in einem
strukturierten Dialog an. Dies hilft dir, Probleme früh zu erkennen, bevor sie
zu verdorbenem Material führen.

![Plausibilitätsprüfungs-Dialog](/screenshots/sanity-check.png)

## Durchgeführte Prüfungen

- **Maschinengrenzverletzungen**: Geometrie, die über das physisch
  erreichbare Maß deiner Maschine hinausgeht, pro Achse und Richtung gemeldet
- **Arbeitsbereichverletzungen**: Werkstücke außerhalb der konfigurierten
  Arbeitsbereichgrenzen
- **No-Go-Zonen-Kollisionen**: Werkzeugwege, die durch aktivierte No-Go-Zonen
  verlaufen

Jede Prüfung erzeugt höchstens ein Problem pro eindeutigem Verstoß, sodass der
Dialog auch bei komplexen Projekten übersichtlich bleibt. Der Dialog
unterscheidet zwischen Fehlern und Warnungen, und du kannst alles überprüfen,
bevor du entscheidest, ob du fortfahren möchtest.

---

## Verwandte Seiten

- [No-Go-Zonen](../machine/nogo-zones) - Eingeschränkte Bereiche auf der
  Arbeitsfläche definieren
- [3D-Ansicht](../ui/3d-preview) - 3D-Werkzeugweg-Visualisierung
