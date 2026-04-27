# Deepnest

Deepnest ordnet deine Werkstücke automatisch in einem kompakten Layout auf
deinem Material oder Maschinenarbeitsbereich an. Es verwendet einen
genetischen Algorithmus, um eine effiziente Packung der Formen zu finden,
Abfall zu minimieren und mehr Teile auf jedes Blech zu bringen.

![Deepnest-Einstellungsdialog](/screenshots/addon-deepnest.png)

## Voraussetzungen

Wähle ein oder mehrere Werkstücke auf der Leinwand aus, bevor du das Nesting
startest. Du kannst auch Material-Elemente auswählen, um die
Blechgrenzen zu definieren. Wenn kein Material ausgewählt ist, verwendet das
Addon das Dokument-Material oder greift auf den Maschinenarbeitsbereich
zurück.

## Nesting-Layout starten

Starte das Nesting-Layout über das Menü **Anordnen**, die
Werkzeugleistenschaltfläche oder das Tastenkürzel **Strg+Alt+N**. Ein
Einstellungsdialog öffnet sich, bevor der Algorithmus ausgeführt wird.

## Nesting-Einstellungen

Der Einstellungsdialog bietet die folgenden Optionen, bevor der
Nesting-Algorithmus beginnt.

**Abstand** legt den Abstand zwischen den verschachtelten Formen in
Millimetern fest. Der Standardwert wird aus der Laserspotgröße deiner
Maschine übernommen. Erhöhe diesen Wert, um einen Sicherheitsabstand zwischen
den Teilen hinzuzufügen.

**Drehung einschränken** behält alle Teile in ihrer ursprünglichen
Ausrichtung. Wenn diese Option deaktiviert ist, dreht der Algorithmus die
Teile in 10-Grad-Schritten, um eine engere Passform zu finden. Freie
Drehung führt zu besserer Materialausnutzung, dauert aber länger.

**Horizontales Spiegeln erlauben** spiegelt Teile während des Nestings
horizontal. Dies kann helfen, Teile enger anzuordnen, jedoch werden die
resultierenden Schnitte gespiegelt.

**Vertikales Spiegeln erlauben** spiegelt Teile während des Nestings
vertikal. Die gleiche Überlegung zur gespiegelten Ausgabe gilt hier.

Klicke auf **Nesting starten**, um zu beginnen. Der Dialog schließt sich und
der Algorithmus läuft im Hintergrund. Eine Fortschrittsanzeige erscheint im
unteren Bedienfeld, während das Nesting läuft.

## Nach dem Nesting

Wenn der Algorithmus abgeschlossen ist, werden alle Werkstücke auf der
Leinwand an ihre verschachtelten Positionen verschoben. Die Positionen werden
als einzelne rücknehmbare Aktion angewendet, sodass du das Layout mit einem
Schritt rückgängig machen kannst, wenn das Ergebnis nicht deinen
Vorstellungen entspricht.

Wenn der Algorithmus nicht alle Werkstücke auf das verfügbare Material
platzieren konnte, werden die nicht platzierten Elemente rechts neben das
Material verschoben, damit sie sichtbar und leicht zu erkennen bleiben.

Wenn das Nesting-Ergebnis schlechter ist als das ursprüngliche Layout — zum
Beispiel, wenn die Teile bereits gut passen — bleiben die Werkstücke in ihren
ursprünglichen Positionen.

## Verwandte Themen

- [Material-Handhabung](../features/stock-handling) - Material für das Nesting definieren
- [Werkstückpositionierung](../features/workpiece-positioning) - Werkstücke manuell positionieren
