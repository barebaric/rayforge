# Rotationsachse einrichten

Rayforge unterstützt Rotationsvorsätze zum Gravieren und Schneiden zylindrischer
Objekte wie Becher, Gläser, Stifte und Rundmaterial. Wenn ein Rotationsmodul
verbunden ist, wickelt Rayforge den Auftrag um den Zylinder und zeigt eine 3D-Vorschau
des Ergebnisses an.

![Rotationsmodul-Einstellungen](/screenshots/machine-rotary-module.png)

## Wann du den Rotationsmodus brauchst

Verwende den Rotationsmodus immer dann, wenn dein Werkstück zylindrisch ist.
Typische Anwendungsfälle sind:

- Gravieren von Logos oder Text auf Trinkgefäßen
- Schneiden von Mustern auf Rohren oder Leitungen
- Markieren zylindrischer Objekte wie Stifte oder Werkzeuggriffe

Ohne Rotationsmodus bewegt die Y-Achse den Laserkopf vor und zurück auf einer
flachen Arbeitsfläche. Mit aktiviertem Rotationsmodus steuert die Y-Achse stattdessen
die Drehung des Zylinders, sodass sich das Design um die Oberfläche wickelt.

## Rotationsmodul einrichten

Verbinde zuerst das Rotationsmodul physisch mit der Maschine, gemäß den Anweisungen
des Herstellers. Normalerweise wird es an den Stepper-Treiber-Anschluss der Y-Achse
anstelle des normalen Y-Achsen-Motors angeschlossen.

Öffne in Rayforge **Einstellungen → Maschine** und navigiere zur Seite **Rotation**,
um dein Modul zu konfigurieren:

- **Umfang**: Miss den Abstand um das Objekt, das du gravieren möchtest. Du kannst
  ein Stück Papier oder Schnur um den Zylinder wickeln und seine Länge messen. Das
  sagt Rayforge, wie groß die zylindrische Oberfläche ist, damit das Design korrekt
  skaliert wird.
- **Mikroschritte pro Umdrehung**: Dies ist die Anzahl der Schritte, die der
  Rotationsmotor für eine volle Umdrehung benötigt. Lies in der Dokumentation deines
  Rotationsmoduls nach, um diesen Wert zu finden.

### Rotationsmodi

Rayforge unterstützt zwei Rotationsmodi:

- **Echte 4. Achse**: Die Rotation arbeitet als unabhängige vierte Achse neben X, Y
  und Z. Dies ist der bevorzugte Modus, wenn dein Controller ihn unterstützt.
- **Achsenersetzung**: Die Rotation übernimmt die Y- oder Z-Achse. Dies ist der
  traditionelle Modus, der von den meisten Hobby-Controllern verwendet wird, bei denen
  das Rotationsmodul an einen vorhandenen Stepper-Treiber-Anschluss angeschlossen wird.

Du kannst den Modus auf der Einstellungsseite Rotation auswählen.

### Rollen-Rotationsvorsätze

Rollen-Rotationsvorsätze (bei denen das Objekt auf Rollen anstatt eingespannt ruht)
haben ihre eigene Einstellungsseite. Wenn deine Rotation Rollen anstelle eines Futters
verwendet, wähle den Rollentyp in der Rotationskonfiguration und gib die Rollenparameter ein.

## Rotationsmodus pro Ebene

Wenn dein Dokument mehrere Ebenen hat, kannst du den Rotationsmodus für jede Ebene
unabhängig aktivieren oder deaktivieren. Das ist nützlich, wenn du flache und
zylindrische Bearbeitung in einem einzigen Projekt kombinieren möchtest oder
unterschiedliche Rotationseinstellungen für verschiedene Teile des Auftrags brauchst.

Wenn der Rotationsmodus auf einer Ebene aktiv ist, erscheint ein kleines Rotations-
Symbol neben der Ebene in der Ebenenliste, damit du auf einen Blick erkennen kannst,
welche Ebenen im Rotationsmodus ausgeführt werden.

## 3D-Vorschau im Rotationsmodus

Wenn der Rotationsmodus aktiv ist, zeigt die [3D-Ansicht](../ui/3d-preview) deinen
Werkzeugpfad um einen Zylinder gewickelt statt auf einer flachen Oberfläche. Die 2D-Arbeitsfläche passt sich ebenfalls automatisch dem Rotationsmodus an, und die 3D-Ansicht gibt Rotationswerkzeugpfade in jeder Konfiguration genau wieder — einschließlich während der Simulationswiedergabe.

![3D-Vorschau im Rotationsmodus](/screenshots/main-3d-rotary.png)

Dies gibt dir eine realistische Vorschau, wie das Design auf dem tatsächlichen Objekt
aussehen wird, und macht es einfacher, Größen- oder Platzierungsprobleme zu erkennen,
bevor du mit dem Schneiden beginnst.

### Rotationsmodul 3D-Modell

Jedem Rotationsmodul kann ein 3D-Modell zugewiesen werden. Das Modell erscheint in
der [3D-Ansicht](../ui/3d-preview) neben deinem Werkzeugpfad und gibt dir einen
besseren Eindruck davon, wie das physische Setup aussieht. Du kannst Skalierung,
Position und Rotation des Modells an deine tatsächliche Hardware anpassen.

### Z-Abtrag

Der Rotationsmodus unterstützt Z-Abtrag, sodass du auf zylindrischen Werkstücken
über mehrere Durchgänge progressiv tiefer schneiden kannst.

## Tipps für gute Ergebnisse

- **Messe den Umfang sorgfältig** — selbst ein kleiner Fehler hier wird dein Design
  auf dem Zylinder stauchen oder strecken.
- **Befestige das Werkstück** — stelle sicher, dass das Objekt fest auf den Rollen
  sitzt und während des Auftrags nicht wackelt oder verrutscht.
- **Teste zuerst mit niedriger Leistung** — führe einen leichten Gravurdurchlauf
  durch, um die Ausrichtung zu überprüfen, bevor du mit voller Leistung schneidest.
- **Halte die Oberfläche sauber** — Staub oder Rückstände auf dem Zylinder können
  die Gravurqualität beeinträchtigen.

## Verwandte Seiten

- [Mehrebenen-Workflow](../features/multi-layer) - Ebenen-Einstellungen inklusive Rotation
- [3D-Ansicht](../ui/3d-preview) - Werkzeugpfade in 3D vorschauen
- [Maschineneinstellungen](general) - Allgemeine Maschinenkonfiguration
