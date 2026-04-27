# Modo Projetor

O Modo Projetor exibe sua área de corte em uma janela separada, projetada
para ser mostrada em um projetor externo ou monitor secundário. Isso permite
ver exatamente onde o laser vai cortar, projetando as trajetórias diretamente
sobre o material, facilitando o alinhamento.

A janela do projetor mostra seus workpieces renderizados em verde brilhante
sobre um fundo preto. Ela exibe o quadro de extensão dos eixos da máquina e a
origem de trabalho para que você possa ver toda a área de corte e onde está o
ponto de origem. A visualização é atualizada em tempo real conforme você move
ou modifica workpieces na tela principal.

## Abrindo a janela do projetor

Abra a janela do projetor em **Visualizar - Mostrar diálogo do projetor**. A
janela abre como uma janela separada e independente que você pode arrastar
para qualquer tela conectada ao seu sistema.

Um interruptor controla a janela do projetor — o mesmo item de menu a fecha, e
pressionar Escape enquanto a janela do projetor está em foco também a fecha.

## Modo tela cheia

Clique no botão **Tela cheia** na barra de título da janela do projetor para
entrar no modo tela cheia. Isso oculta as decorações da janela e preenche
toda a tela. Clique em **Sair da tela cheia** (o mesmo botão) para retornar
ao modo de janela.

A tela cheia é o modo pretendido ao projetar sobre o material, pois remove o
contorno de janela distrativo e utiliza toda a superfície da tela.

## Opacidade

O botão de opacidade na barra de título alterna entre quatro níveis: 100%,
80%, 60% e 40%. Reduzir a opacidade torna a janela do projetor
semitransparente, o que pode ser útil em um monitor de mesa para ver as
janelas atrás dela. Cada clique avança para o próximo nível de opacidade e
retorna ao início.

![Modo Projetor](/screenshots/addon-projector-mode.png)

## O que o projetor mostra

A exibição do projetor renderiza uma visualização simplificada do seu
documento. Os workpieces aparecem como contornos verde brilhantes mostrando as
trajetórias calculadas — os mesmos caminhos que serão enviados ao laser. As
imagens base dos seus workpieces não são exibidas, mantendo a exibição
focada nas trajetórias de corte.

O quadro de extensão da máquina aparece como uma borda representando toda a
área de deslocamento dos eixos da sua máquina. A mira da origem de trabalho
mostra onde a origem do sistema de coordenadas está localizada dentro dessa
área. Ambos são atualizados automaticamente se você alterar o deslocamento do
sistema de coordenadas de trabalho na sua máquina.

## Tópicos relacionados

- [Sistemas de coordenadas](../general-info/coordinate-systems) - Entender coordenadas da máquina e deslocamentos de trabalho
- [Posicionamento de workpieces](../features/workpiece-positioning) - Posicionar workpieces na tela
