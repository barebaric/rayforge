---
description:
  "Sondeie a superfície de trabalho do seu laser em uma grade e compense automaticamente uma mesa
  irregular. Mantém o ponto focal sobre o material em mesas grandes ou onduladas."
---

# Malha da mesa

Grandes superfícies de trabalho raramente são perfeitamente planas. Quando a altura da mesa varia
mais do que a profundidade de foco do seu laser, os cortes ficam inconsistentes ao longo do
material. O recurso de malha da mesa mede a altura da superfície em uma grade e compensa o eixo Z do
trajeto para que o ponto focal siga a superfície real.

![Malha da mesa](/screenshots/machine-settings-bed-mesh.webp)

## Requisitos

A malha da mesa requer uma máquina com eixo Z e um driver que suporte sonda (GRBL, Marlin, Smoothie
e OctoPrint atualmente). A página **Malha da mesa** só aparece em **Configurações → Máquina** quando
ambos estão presentes. Ela fica após a página do dispositivo.

## Medir a mesa

Abra **Configurações → Máquina** e navegue até a página **Malha da mesa**.

1. **Grade de medição**: Defina a área a ser medida (origem X/Y, largura, altura) e a densidade da
   grade (colunas e linhas). A página mostra o número de pontos de medição e estima a duração.
   Grades mais densas seguem a superfície com mais precisão, mas levam mais tempo.
2. **Medição**: Configure a velocidade de avanço da medição, o quanto a cabeça pode descer
   procurando a superfície em cada ponto (Curso máximo) e a altura Z segura usada para mover entre
   os pontos.
3. Clique em **Iniciar medição**. A máquina visita cada ponto da grade em padrão serpenteante e toca
   a superfície em cada um; a visualização 3D é preenchida ao vivo conforme os resultados chegam.
   Você pode interromper a execução a qualquer momento; a malha só é salva quando a grade completa
   termina.

A malha é armazenada junto com o perfil da máquina e exibida como uma superfície 3D colorida: as
áreas azuis são mais baixas e as vermelhas mais altas. Use o botão **Excluir malha** para removê- la
se não quiser mais compensação de altura.

:::note Durante a medição, a cabeça se move por toda a área da grade. Limpe a mesa de objetos que
possam bloquear a sonda e certifique-se de que a ponta da sonda (ou a mira do laser) alcance a
superfície em todos os pontos da grade. :::

---

## Páginas relacionadas

- [Configurações de hardware](hardware) - Dimensões da máquina e configuração dos eixos
- [Configurações do dispositivo](device) - Conexão e opções do controlador
- [Visualização 3D](../ui/3d-preview.md) - Visualização 3D do trajeto
