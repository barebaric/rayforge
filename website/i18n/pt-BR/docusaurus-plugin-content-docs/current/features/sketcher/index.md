---
description: "Use o esboçador paramétrico 2D integrado do Rayforge para criar designs laser prontos com linhas, círculos, curvas de Bézier e restrições."
---

# Esboçador paramétrico 2D

O Esboçador paramétrico 2D é um recurso poderoso do Rayforge que permite criar
e editar designs 2D precisos baseados em restrições diretamente no aplicativo.
Esse recurso permite projetar peças personalizadas do zero sem precisar de
software CAD externo.

## Visão geral

O esboçador fornece um conjunto completo de ferramentas para criar formas
geométricas e aplicar restrições paramétricas para definir relações precisas
entre os elementos. Essa abordagem garante que seus designs mantenham a
geometria pretendida mesmo quando as dimensões são modificadas.

## Criando e editando esboços

### Criando um novo esboço

1. Abra o painel inferior e clique no botão **Novo esboço**, ou clique com o
   botão direito no canvas e selecione **Novo esboço** no menu de contexto.
2. Um novo espaço de trabalho vazio será aberto com a interface do editor de
   esboços
3. Comece a criar geometria usando as ferramentas de desenho do menu circular
   ou os atalhos de teclado
4. Aplique restrições para definir as relações entre os elementos
5. Clique em "Finalizar esboço" para salvar seu trabalho e retornar ao espaço
   de trabalho principal

### Editando esboços existentes

1. Dê um duplo clique em uma peça baseada em esboço no espaço de trabalho
   principal
2. Alternativamente, selecione um esboço e escolha "Editar esboço" no menu de
   contexto
3. Faça suas modificações usando as mesmas ferramentas e restrições
4. Clique em "Finalizar esboço" para salvar as alterações ou em "Cancelar
   esboço" para descartá-las

## Dicas de fluxo de trabalho

1. **Comece com geometria aproximada**: Crie formas básicas primeiro e depois
   refine com restrições
2. **Use restrições cedo**: Aplique restrições enquanto constrói para manter a
   intenção do design
3. **Verifique o status das restrições**: O sistema indica quando os esboços
   estão totalmente restritos
4. **Fique atento a conflitos**: Restrições que conflitam entre si são
   destacadas em vermelho e mostradas no painel de restrições para fácil
   identificação
5. **Utilize a simetria**: Restrições de simetria podem acelerar
   significativamente designs complexos
6. **Use a grade**: Ative a grade para alinhamento preciso e use Ctrl para
   snapar na grade
7. **Itere e refine**: Não hesite em modificar restrições para obter o
   resultado desejado

## Recursos de edição

- **Suporte completo a desfazer/refazer**: O estado completo do esboço é salvo
  com cada operação
- **Cursor dinâmico**: O cursor muda para refletir a ferramenta de desenho
  ativa
- **Visualização de restrições**: As restrições aplicadas são claramente
  indicadas na interface
- **Atualizações em tempo real**: Alterações nas restrições atualizam
  imediatamente a geometria
- **Edição por duplo clique**: Dar um duplo clique em restrições dimensionais
  (Distância, Raio, Diâmetro, Ângulo, Proporção) abre um diálogo para editar
  seus valores
- **Expressões paramétricas**: Restrições dimensionais suportam expressões,
  permitindo que valores sejam calculados a partir de outros parâmetros (por
  ex., `width/2` para um raio que seja metade da largura)
