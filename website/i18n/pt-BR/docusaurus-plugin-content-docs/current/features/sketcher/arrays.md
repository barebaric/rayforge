---
description: "Crie matrizes circulares e matrizes ao longo de curvas no esboçador paramétrico do Rayforge."
---

# Matrizes

O esboçador fornece duas ferramentas de matriz para criar matrizes
paramétricas: **Matriz Circular** e **Matriz ao longo de curva**.

## Matrizes Circulares

A ferramenta **Matriz Circular** (`G+Y`) cria um padrão polar paramétrico
a partir da seleção atual:

1. Selecione as entidades que deseja padronizar.
2. Ative a ferramenta pela barra de ferramentas, pelo menu
   **Esboço → Matrizes** ou `G+Y`.
3. Um círculo guia aparece na tela e um diálogo não modal é aberto com
   uma pré-visualização ao vivo.
4. Defina a **quantidade** e o **ângulo total**. As cópias são geradas
   parametricamente ao redor do centro do círculo guia.
5. Arraste o centro do círculo guia para reposicionar a matriz, ou
   arraste a entidade original para alterar o raio — os campos do
   diálogo são atualizados ao vivo.
6. A **dimensão de raio** do círculo guia redimensiona toda a matriz.
   Dê um **duplo clique** no círculo guia para reabrir o diálogo de
   edição e regenerar membros ausentes ou redistribuir.

As cópias são geometria estática assada sem restrições do resolvedor:
elas são regeneradas a partir do modelo quando a matriz é editada.
Excluir um membro remove apenas a geometria desse membro e nunca
redistribui os sobreviventes.

## Matriz ao longo de curva

A ferramenta **Matriz ao longo de curva** distribui cópias de uma ou mais
entidades ao longo de um caminho guia (uma linha, arco ou curva de Bézier).
As cópias são colocadas diretamente no caminho e seguem sua tangente em cada
posição.

### Criando uma matriz ao longo de curva

1. Desenhe a forma que deseja distribuir (a semente) e o caminho guia que
   deseja seguir.
2. Selecione ambos: primeiro clique no **caminho guia**, depois
   Shift-clique nas **entidades semente**.
3. Ative a ferramenta pela barra de ferramentas, pelo menu
   **Esboço → Matrizes** ou `G+W`.
4. Um diálogo não modal abre mostrando uma pré-visualização ao vivo com
   cópias distribuídas ao longo do caminho.
5. Ajuste a **quantidade** (total de membros incluindo o modelo no início
   do caminho) ou defina um valor de **espaçamento** para derivar a
   quantidade automaticamente do comprimento do caminho.
6. Opcionalmente habilite **Alinhar à tangente** para que cada cópia gire
   para seguir a direção do caminho em sua posição.
7. Use **Deslocamento do início** para pular uma seção inicial do caminho
   antes de colocar a primeira cópia.

### Editando uma matriz ao longo de curva

- Dê um **duplo clique** no caminho guia (ou clique em **Editar** na barra
  de ferramentas) para reabrir o diálogo e alterar quantidade, espaçamento,
  deslocamento ou configurações de alinhamento.
- **Arraste** qualquer extremidade do caminho guia para remodelá-lo. Quando
  soltar, todas as cópias são automaticamente redistribuídas ao longo da
  nova geometria do caminho — incluindo atualizações de rotação quando
  *Alinhar à tangente* está habilitado.
- A forma semente pode ser editada como qualquer outra geometria do esboço;
  alterações se propagam para todas as cópias na próxima atualização.

### Como funciona

As cópias são geometria estática assada — não estão vinculadas ao modelo
através de restrições do resolvedor. Quando o caminho guia é editado,
`sync_arrays` detecta a mudança e regenera todas as cópias do zero
usando a geometria atual do caminho. Isso mantém as atualizações rápidas
e evita sobrecarga do resolvedor.

O modelo (slot 0) é colocado no início do caminho. Sua posição e orientação
são atualizadas automaticamente quando o caminho é editado. As entidades
semente originais são removidas quando a matriz é criada; desfazer as
restaura.
