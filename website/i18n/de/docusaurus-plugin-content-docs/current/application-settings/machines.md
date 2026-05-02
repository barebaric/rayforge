---
description: "Verwalte Maschinen in Rayforge — füge sie hinzu, konfiguriere, exportiere, importiere und wechsle zwischen verschiedenen Laserschneidern und Gravierern für deine Projekte."
---

# Maschinen

![Maschinen-Einstellungen](/screenshots/application-machines.png)

Die Maschinen-Seite in den Anwendungseinstellungen zeigt eine Liste aller
konfigurierten Maschinen. Jeder Eintrag zeigt den Maschinennamen und hat
Schaltflächen zum Bearbeiten oder Löschen. Die aktuell aktive Maschine ist
mit einem Häkchen markiert.

## Eine Maschine hinzufügen

1. Klicke auf die Schaltfläche **Add Machine** am unteren Ende der Liste
2. Wähle ein Geräteprofil aus der Liste als Vorlage — jedes Profil
   konfiguriert die Maschineneinstellungen und den G-Code-Dialekt vor

![Maschine hinzufügen](/screenshots/add-machine-dialog.png)

3. Der [Maschineneinstellungen-Dialog](../machine/general) öffnet sich, in
   dem du die Konfiguration anpassen kannst

Alternativ kannst du auf **Import from File...** im Profil-Auswahldialog
klicken, um eine Maschine aus einem zuvor exportierten Profil hinzuzufügen.

## Eine Maschine bearbeiten

Klicke auf das Bearbeiten-Symbol neben einer Maschine, um den
[Maschineneinstellungen-Dialog](../machine/general) zu öffnen.

## Aktive Maschine wechseln

Verwende das Maschinen-Dropdown in der Kopfzeile des Hauptfensters, um
zwischen konfigurierten Maschinen zu wechseln. Die Auswahl wird zwischen
Sitzungen gespeichert.

## Eine Maschine löschen

1. Klicke auf das Lösch-Symbol neben der Maschine
2. Bestätige das Löschen

:::warning
Das Löschen einer Maschine kann nicht rückgängig gemacht werden. Exportiere
das Profil zuerst, wenn du die Konfiguration behalten möchtest.
:::
