---
description:
  "Corte vinil, papelão, couro e outros materiais em folha com uma faca de arrasto ou uma faca
  tangencial motorizada em vez de um laser."
---

# Corte com faca

O Rayforge também pode acionar suportes de faca além de lasers. Dois passos integrados cortam
contornos vetoriais com uma faca: **Corte com faca de arrasto** para um suporte de lâmina de arrasto
e **Corte com faca tangencial** para uma lâmina rotativa motorizada. Os passos de faca só aparecem
em máquinas com um cabeçote de faca compatível, então uma máquina pode misturar operações de laser e
de faca no mesmo projeto.

## Requisitos

- Uma máquina com um cabeçote **Faca de arrasto** ou **Faca tangencial**. Adicione o cabeçote em
  [Configurações da máquina → Cabeçotes](../machine/general.md) com **Adicionar cabeçote**.
- Um controlador G-code capaz de mover a máquina sem laser. Os passos de faca cortam na linha
  central e não usam potência de laser.
- O complemento Drag Knife integrado. Ele acompanha o Rayforge; se foi desativado, ative-o em
  **Configurações → Complementos**.

## Configurar um cabeçote de faca

1. Abra **Máquina → Configurações atuais da máquina → Cabeçotes**.
2. Clique em **Adicionar cabeçote** e escolha **Faca de arrasto** ou **Faca tangencial**.
3. Dê um nome ao cabeçote. Para uma faca de arrasto, informe o **Deslocamento da lâmina**: a
   distância entre a ponta da lâmina e o ponto de giro da máquina.
4. Feche as configurações. Os passos de faca correspondentes agora são oferecidos ao adicionar uma
   operação.

O cabeçote de faca também é uma capacidade da máquina: operações que precisam de uma faca só ficam
disponíveis enquanto a máquina ativa tiver um.

## Corte com faca de arrasto

Uma faca de arrasto tem uma lâmina que segue atrás do ponto de giro. O Rayforge compensa o
deslocamento automaticamente, para que a lâmina siga o contorno em vez do ponto de giro.

1. Selecione os contornos a cortar.
2. Adicione uma operação **Corte com faca de arrasto**.
3. Escolha o cabeçote **Faca** e ajuste **Deslocamento da lâmina**, **Altura Z segura**, **Ângulo de
   giro** e a velocidade de corte.

O passo reescreve cada contorno no caminho que a máquina deve seguir: um arco de entrada alinha a
lâmina com a primeira direção de corte, segmentos retos são deslocados para a frente pelo
deslocamento da lâmina, cantos dentro do ângulo de giro são girados com a lâmina abaixada, cantos
mais fechados levantam a lâmina para girar no ar, e um arco de saída corta além do fim do caminho.

Ajuste o **Deslocamento da lâmina** conforme a documentação do suporte ou meça-o. Um valor muito
pequeno deixa os cantos arredondados; um valor muito grande passa deles. O **Ângulo de giro** limita
quanto a lâmina gira abaixada; cantos mais fechados levantam a faca primeiro para que a lâmina não
arraste de lado.

## Corte com faca tangencial

Uma faca tangencial tem um motor que gira a lâmina para mantê-la tangente ao caminho. O passo
adiciona a direção do caminho como valor do eixo **A** a cada movimento de corte, para que o
controlador possa girar a lâmina.

1. Selecione os contornos a cortar.
2. Adicione uma operação **Corte com faca tangencial**.
3. Escolha o cabeçote **Faca** e ajuste **Tolerância de ângulo**, **Tolerância de raio**, **RPM do
   spindle**, **Altura Z segura** e a velocidade de corte.

Mudanças de direção maiores que a **Tolerância de ângulo** ou arcos mais fechados que a **Tolerância
de raio** levantam a faca até a Z segura, giram-na e mergulham de volta. Mudanças menores giram a
lâmina no lugar com a faca abaixada. O **RPM do spindle** é emitido como comando M3 para o motor da
lâmina; o Rayforge nunca usa M4, porque o motor da lâmina só gira no sentido horário.

## Dicas

- Execute o trabalho primeiro em velocidade reduzida, especialmente em material espesso ou fibroso.
- Use uma base de corte sacrificial ou uma profundidade pequena para a lâmina não marcar a mesa.
- Os passos de faca são operações G-code normais: podem ser combinados com passos de laser,
  reordenados, duplicados e salvos em receitas como qualquer outra operação.
