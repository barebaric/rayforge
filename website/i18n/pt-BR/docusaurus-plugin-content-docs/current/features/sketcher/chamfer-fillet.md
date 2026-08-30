---
description:
  "Chanfre cantos agudos com a ferramenta de chanfro ou arredonde-os com a ferramenta de
  arredondamento no esboçador do Rayforge."
---

# Chanfro e arredondamento

O esboçador fornece duas ferramentas para modificar cantos onde duas linhas se encontram:

- **Chanfro** (`C+H`): substitui um canto agudo por uma borda chanfrada.
- **Arredondamento** (`C+F`): substitui um canto agudo por uma borda arredondada.

![Um retângulo chanfrado ao lado de um retângulo arredondado](/screenshots/addons-sketcher-tool-chamfer-fillet.webp)

Para aplicar uma delas:

1. Selecione um ponto de junção onde exatamente duas linhas se encontram.
2. Pressione `C+H` para chanfro ou `C+F` para arredondamento, ou escolha a ferramenta no menu
   circular.

O canto é substituído em uma única etapa. As duas linhas são aparadas de volta e a nova borda é
inserida entre elas, junto com restrições que mantêm os segmentos aparados colineares com os
originais e o canto simétrico. Em um chanfro, o comprimento do chanfro assume por padrão uma fração
da linha adjacente mais curta; em um arredondamento, o raio do arco é escolhido para se ajustar.
Arrastar as extremidades da borda inserida depois ajusta seu tamanho, com as restrições mantendo o
canto intacto.
