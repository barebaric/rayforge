# Enviando Alterações

Este guia cobre o processo para contribuir com melhorias de código ao Rayforge.

## Crie uma Branch de Feature

Crie uma branch descritiva para suas alterações:

```bash
git checkout -b feature/seu-nome-de-feature
# ou
git checkout -b fix/numero-issue-descricao
```

## Faça Suas Alterações

- Siga o estilo e convenções de código existentes
- Escreva commits limpos e focados com mensagens claras
- Adicione testes para novas funcionalidades
- Atualize documentação conforme necessário

## Teste Suas Alterações

Execute a suíte de testes completa para garantir que nada está quebrado:

```bash
# Execute todos os testes e linting
pixi run test
pixi run lint
```

## Sincronize com Upstream

Antes de criar um pull request, sincronize com o repositório upstream:

```bash
# Busque as últimas alterações
git fetch upstream

# Faça rebase da sua branch no main mais recente
git rebase upstream/main
```

## Envie um Pull Request

1. Envie sua branch para seu fork:
   ```bash
   git push origin feature/seu-nome-de-feature
   ```

2. Crie um pull request no GitHub com:
   - Um título claro descrevendo a mudança
   - Uma descrição detalhada do que você mudou e por quê
   - Referência a quaisquer issues relacionados
   - Screenshots se a mudança afeta a UI

## Processo de Code Review

- Todos os pull requests requerem revisão antes de merge
- Responda feedback prontamente e faça as alterações solicitadas
- Mantenha a discussão focada e construtiva

## Requisitos de Merge

Pull requests são merged quando:

- [ ] Passam todos os testes automatizados
- [ ] Seguem o estilo de codificação do projeto
- [ ] Incluem testes apropriados para nova funcionalidade
- [ ] Têm atualizações de documentação se necessário
- [ ] São aprovados por pelo menos um mantenedor

## Diretrizes Adicionais

### Mensagens de Commit

Use mensagens de commit claras e descritivas:

- Comece com letra maiúscula
- Mantenha a primeira linha sob 50 caracteres
- Use o modo imperativo ("Adiciona feature" não "Adicionou feature")
- Inclua mais detalhes no corpo se necessário

### Alterações Pequenas e Focadas

Mantenha pull requests focados em uma única feature ou correção. Alterações grandes devem ser divididas em partes menores e lógicas.

:::tip Discuta Primeiro
Para mudanças maiores, abra uma [issue](https://github.com/barebaric/rayforge/issues) primeiro para discutir sua abordagem antes de investir tempo significativo.
:::


:::note Precisa de Ajuda?
Se você não tiver certeza sobre qualquer parte do processo de contribuição, não hesite em pedir ajuda em uma issue ou discussão.
:::
