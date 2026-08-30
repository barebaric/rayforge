---
description:
  "Fase scharfe Ecken mit dem Fasen-Werkzeug ab oder runde sie mit dem Verrundungs-Werkzeug im
  Rayforge-Sketcher."
---

# Fase und Verrundung

Der Sketcher bietet zwei Werkzeuge, um Ecken zu modifizieren, an denen zwei Linien
aufeinandertreffen:

- **Fase** (`C+H`): ersetzt eine scharfe Ecke durch eine abgeschrägte Kante.
- **Verrundung** (`C+F`): ersetzt eine scharfe Ecke durch eine abgerundete Kante.

![Ein Rechteck mit Fase neben einem Rechteck mit Verrundung](/screenshots/addons-sketcher-tool-chamfer-fillet.webp)

Um eines davon anzuwenden:

1. Wähle einen Verbindungspunkt aus, an dem genau zwei Linien aufeinandertreffen.
2. Drücke `C+H` für eine Fase oder `C+F` für eine Verrundung, oder wähle das Werkzeug aus dem
   Kreismenü.

Die Ecke wird in einem einzigen Schritt ersetzt. Die beiden Linien werden zurückgestutzt und die
neue Kante zwischen ihnen eingefügt, zusammen mit Einschränkungen, die die gestutzten Segmente
kollinear zu den Originalen und die Ecke symmetrisch halten. Bei einer Fase beträgt die Fasenlänge
standardmäßig einen Bruchteil der kürzeren angrenzenden Linie; bei einer Verrundung wird der
Bogenradius passend gewählt. Ziehst du anschließend die Endpunkte der eingefügten Kante, passt sich
ihre Größe an, während die Einschränkungen die Ecke intakt halten.
