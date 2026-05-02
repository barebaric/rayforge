---
description: "Antes de executar ou exportar um trabalho, o Rayforge verifica automaticamente problemas comuns como violações de limites, violações de área de trabalho e colisões com zonas de restrição."
---

# Verificações de Sanidade do Trabalho

Antes de executar ou exportar um trabalho, o Rayforge realiza automaticamente um
conjunto de verificações de sanidade e apresenta os resultados em um diálogo
estruturado. Isso ajuda você a detectar problemas cedo, antes que se tornem
material desperdiçado.

![Diálogo de Verificação de Sanidade](/screenshots/sanity-check.png)

## Verificações Realizadas

- **Violações de limites da máquina**: Geometria que se estende além do que sua
  máquina pode alcançar fisicamente, relatada por eixo e direção
- **Violações de área de trabalho**: Peças de trabalho fora dos limites da área
  de trabalho configurada
- **Colisões com zonas de restrição**: Caminhos de ferramenta que passam por
  zonas de restrição habilitadas

Cada verificação produz no máximo um problema por violação única, mantendo o
diálogo legível mesmo para projetos complexos. O diálogo distingue entre erros
e avisos, e você pode revisar tudo antes de decidir se deve prosseguir.

---

## Páginas Relacionadas

- [Zonas de Restrição](../machine/nogo-zones) - Defina áreas restritas na
  superfície de trabalho
- [Visualização 3D](../ui/3d-preview) - Visualização de caminhos de ferramenta
  em 3D
