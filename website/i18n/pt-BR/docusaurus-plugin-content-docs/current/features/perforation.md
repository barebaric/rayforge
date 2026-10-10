---
description:
  "Corte linhas tracejadas no Rayforge com o pós-processador Perfuração: defina um comprimento de
  corte e um comprimento de salto para dobras, linhas destacáveis e furos de costura."
---

# Perfuração

O pós-processador **Perfuração** transforma um corte contínuo em um corte tracejado. O laser dispara
pelo **Comprimento de corte**, depois se desloca com o laser desligado pelo **Comprimento de
salto**, e repete isso ao longo de cada contorno da peça.

## Quando usar

- Linhas de dobra em cartolina e papelão ondulado
- Linhas destacáveis (ingressos, cupons, embalagens)
- Furos de costura em couro
- Linhas tracejadas decorativas

## Configurações

A perfuração está disponível nas configurações de pós-processamento das operações de **Contorno**.
Ela vem desativada por padrão.

- **Comprimento de corte**: a distância em que o laser dispara antes de cada intervalo.
- **Comprimento de salto**: a distância percorrida com o laser desligado entre dois cortes.

O padrão é medido ao longo de cada contorno da peça e recomeça no início de cada contorno, sempre
com um corte completo. Um contorno menor que um corte mais um salto é cortado inteiro.

## Dicas

- Para uma linha de dobra que não deve atravessar o material, comece com um comprimento de salto
  parecido com o de corte e reduza a potência.
- Comprimentos muito curtos (alguns décimos de milímetro) fazem o laser ligar e desligar
  rapidamente, o que reduz a potência efetiva.
- A perfuração funciona junto com [Abas de fixação](holding-tabs), [Entrada/Saída](lead-in-out) e
  [Múltiplas passadas](multi-pass).

## Páginas relacionadas

- [Corte de contorno](operations/contour) - A operação de corte que usa a perfuração
- [Abas de fixação](holding-tabs) - Intervalos colocados à mão para manter as peças presas
