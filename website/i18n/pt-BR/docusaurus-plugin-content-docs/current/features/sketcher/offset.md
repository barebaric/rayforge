---
description: "Aumente, reduza ou transforme trajetos em ranhuras com a ferramenta de deslocamento no esboçador do Rayforge."
---

# Deslocar contorno

A ferramenta de deslocamento (`O+F`) aumenta ou reduz o contorno
selecionado em uma distância informada, ou expande um trajeto aberto em
uma ranhura. Selecione as entidades que formam um contorno (ou use duplo
clique para selecionar a geometria conectada) e pressione `O+F`, ou use
a entrada **Deslocar** no menu circular.

![Diálogo de deslocar contorno](/screenshots/addons-sketcher-offset-dialog.webp)

O diálogo pede a distância de deslocamento e mostra uma pré-visualização
ao vivo do resultado na tela enquanto você digita:

- **Contornos fechados** crescem com distância positiva e encolhem com
  distância negativa. Um deslocamento que colapsaria o contorno é
  recusado.
- **Trajetos abertos** se tornam um contorno fechado em forma de ranhura
  da largura informada, com pontas arredondadas.

![Contorno Bézier](/screenshots/addons-sketcher-offset-before.webp)
![Bézier deslocado em uma ranhura](/screenshots/addons-sketcher-offset-after.webp)

Ao deslocar, o contorno selecionado é substituído pelo resultado:

- Círculos, arcos e elipses isolados mantêm seu tipo de entidade e são
  atualizados no lugar, permanecendo editáveis e restrigíveis como
  antes.
- Cadeias de segmentos conectados (incluindo Béziers) são substituídas
  por uma entidade polígono. O polígono é editado como um todo: arraste
  o ponto central para movê-lo e o ponto de alça para rotacioná-lo ou
  escalá-lo de forma uniforme.

Se a seleção contiver vários contornos desconectados, cada um é
deslocado independentemente em uma única etapa.
