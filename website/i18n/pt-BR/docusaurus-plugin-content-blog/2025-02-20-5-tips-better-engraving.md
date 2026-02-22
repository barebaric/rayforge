---
slug: 5-tips-better-engraving
title: 5 Dicas para Melhores Resultados de Gravação a Laser com Rayforge
authors: rayforge_team
tags: [engraving, optimization, quality, workflow]
---

![Visualização 3D](/screenshots/main-3d.png)

Obter resultados de gravação a laser de qualidade profissional requer mais
do que apenas bom hardware — suas configurações de software e fluxo de
trabalho também importam. Aqui estão cinco dicas para ajudá-lo a aproveitar
ao máximo o Rayforge.

<!-- truncate -->

## 1. Use Overscan para Gravação Raster mais Suave

Ao fazer gravação raster, um problema comum são linhas visíveis ou
inconsistências nas bordas onde o laser muda de direção. Isso acontece
porque a cabeça do laser precisa desacelerar e acelerar, o que pode afetar
a qualidade da gravação.

**Solução**: Habilite **Overscan** nas configurações da sua operação raster.

O overscan estende o trajeto de deslocamento do laser além da área de
gravação real, permitindo que a cabeça atinja a velocidade máxima antes de
entrar na área de trabalho e mantenha essa velocidade durante todo o
percurso. Isso resulta em uma gravação muito mais suave e consistente.

Para habilitar o overscan:

1. Selecione sua operação raster
2. Abra as configurações da operação
3. Habilite "Overscan" e defina a distância (tipicamente 3-5mm funciona bem)

Saiba mais em nosso [guia de Overscan](/docs/features/overscan).

## 2. Otimize o Tempo de Deslocamento com Ordenação de Trajetórias

Para operações de contorno com muitos trajetos separados, a ordem em que
o laser visita cada forma pode impactar significativamente o tempo total
do trabalho.

**Solução**: Use a **otimização de tempo de deslocamento** integrada do
Rayforge.

O Rayforge pode reordenar automaticamente os trajetos para minimizar o
tempo de deslocamento sem corte. Isso é especialmente útil para trabalhos
com muitos objetos pequenos ou texto com múltiplas letras.

A otimização de trajeto é tipicamente habilitada por padrão, mas você pode
verificar e ajustá-la nas configurações da operação de Contorno.

## 3. Adicione Abas de Fixação para Prevenir Movimento da Peça

Nada é mais frustrante do que ter um trabalho de corte quase finalizado
arruinado porque a peça deslocou ou caiu através da mesa da máquina no
último momento.

**Solução**: Use **Abas de Fixação** para manter as peças no lugar até que
o trabalho seja concluído.

As abas de fixação são pequenas seções não cortadas que mantêm sua peça
conectada ao material circundante. Após a conclusão do trabalho, você pode
facilmente remover a peça e limpar as abas com uma faca ou lixa.

O Rayforge suporta colocação de abas tanto manual quanto automática:

- **Manual**: Clique exatamente onde você quer as abas na tela
- **Automática**: Especifique o número de abas e deixe o Rayforge
  distribuí-las uniformemente

Confira a [documentação de Abas de Fixação](/docs/features/holding-tabs)
para um guia completo.

## 4. Visualize Seu Trabalho em 3D Antes de Executar

Um dos recursos mais valiosos do Rayforge é a visualização 3D de G-code.
É tentador pular esta etapa e enviar o trabalho diretamente para a máquina,
mas reservar um momento para visualizar pode economizar tempo e materiais.

**O que procurar na visualização**:

- Verifique se todas as operações estão sendo executadas na ordem correta
- Procure por quaisquer trajetos inesperados ou sobreposições
- Confirme se operações de múltiplas passagens têm o número correto de
  passagens
- Certifique-se de que os limites do trabalho se encaixam dentro do seu
  material

Para abrir a visualização 3D, clique no botão **Visualização 3D** na barra
de ferramentas principal após gerar seu G-code.

Saiba mais sobre a visualização 3D em nossa
[documentação de UI](/docs/ui/3d-preview).

## 5. Use Ganchos de G-code Personalizados para Fluxos de Trabalho Consistentes

Se você se encontra executando os mesmos comandos G-code antes ou depois
de cada trabalho — como homing, ligar um assistente de ar, ou executar uma
rotina de foco — você pode automatizar isso com **Macros e Ganchos de
G-code**.

**Casos de uso comuns**:

- **Gancho pré-trabalho**: Leva a máquina à origem, liga assistente de ar,
  executa uma rotina de auto-foco
- **Gancho pós-trabalho**: Desliga assistente de ar, retorna à posição de
  origem, toca um som de conclusão
- **Macros específicas por camada**: Muda a altura do foco entre operações,
  troca módulos de laser

Os ganchos suportam substituição de variáveis, então você pode referenciar
propriedades do trabalho como espessura do material, tipo de operação, e
mais.

Exemplo de gancho pré-trabalho:

```gcode
G28 ; Home all axes
M8 ; Turn on air assist
G0 Z{focus_height} ; Move to focus height
```

Consulte nosso [guia de Macros e Ganchos de G-code](/docs/machine/hooks-macros)
para exemplos detalhados e referência de variáveis.

---

## Dica Bônus: Teste em Material de Sobra Primeiro

Embora isso não seja específico do Rayforge, vale a pena repetir: sempre
teste novas configurações, operações ou materiais em sobra primeiro. Use
os perfis de material e predefinições de operação do Rayforge para salvar
suas configurações testadas para uso futuro.

---

*Tem suas próprias dicas e truques do Rayforge? Compartilhe com a comunidade
nas [Discussões do GitHub](https://github.com/barebaric/rayforge/discussions)!*
