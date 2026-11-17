---
description:
  "Corte linhas tracejadas no Rayforge com o pós-processador Perfuração: defina um comprimento de
  corte e um comprimento de salto para dobras, linhas destacáveis e furos de costura."
---

# Perfuração

Nem todo corte serve para soltar uma peça do material. Algumas linhas existem para que o material
dobre sobre elas, se rasgue com limpeza ou mostre por onde a agulha vai passar. É para isso que
serve o pós-processador **Perfuração**: ele transforma um corte contínuo em um corte tracejado. Em
vez de manter o laser aceso do começo ao fim, ele dispara pelo **Comprimento de corte**, percorre o
**Comprimento de salto** com o laser desligado e repete esse ritmo ao longo de cada contorno da
peça. O que sobra é uma linha de cortes pequenos que mantém o material unido — até que você decida
dobrá-lo ou rasgá-lo.

![As configurações da Perfuração na página de pós-processamento de uma etapa de Contorno](/screenshots/step-settings-contour-post-perforation.webp)

## Alguns usos típicos

Sempre que uma linha deve enfraquecer o material sem cortá-lo por completo, vale a pena um corte
perfurado. Os casos clássicos:

- **Linhas de dobra** em cartolina e papelão ondulado dobram com precisão sobre um corte tracejado —
  caixas, cartões e designs pop-up, por exemplo.
- **Linhas destacáveis** abrem ingressos, cupons e embalagens sem tesoura.
- **Furos de costura** em couro marcam o caminho da agulha antes de costurar.
- **Contornos tracejados** são, por si só, um efeito decorativo — em madeira, papel e acrílico.

## Ativar a perfuração

A perfuração fica na página **Pós-processamento** das configurações de uma etapa de
[Contorno](operations/contour), junto aos demais pós-processadores. Ela vem desativada por padrão;
ao ligar o interruptor, aparecem dois valores:

- O **Comprimento de corte** é a distância em que o laser permanece aceso antes de cada intervalo.
- O **Comprimento de salto** é a distância percorrida com o laser desligado entre dois cortes.

O padrão recomeça no início de cada contorno, sempre com um corte completo, para que todas as linhas
tracejadas da peça comecem da mesma forma. Um contorno curto demais para um único corte e um salto é
simplesmente cortado inteiro.

## O resultado

Na área de desenho, a linha tracejada substitui o contorno de corte sólido: cada intervalo do
caminho será também um intervalo na peça pronta.

![Um retângulo e um círculo cortados como linhas tracejadas em uma chapa de compensado](/screenshots/operations-perforation.webp)

## Dicas

Para uma linha de dobra que não deve atravessar o material, comece com um comprimento de salto
parecido com o de corte e reduza a potência: um sulco raso dobra tão bem quanto um corte tracejado
profundo, e o verso fica limpo.

Comprimentos muito curtos (alguns décimos de milímetro) fazem o laser ligar e desligar em sequência
rápida, o que reduz a potência efetiva e pode deixar marcas irregulares. Se o padrão aparecer
desigual, experimente valores maiores.

A perfuração convive bem com o resto do pós-processamento: as [Abas de Fixação](holding-tabs), o
[Aproximação / Saída](lead-in-out) e a [Multi-Passagem](multi-pass) continuam funcionando como
sempre.

## Páginas relacionadas

- [Corte de Contorno](operations/contour) — a operação de corte à qual este pós-processador pertence
- [Abas de Fixação](holding-tabs) — intervalos colocados à mão para manter as peças presas
