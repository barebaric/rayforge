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

2. Crie um pull request no GitHub e preencha o modelo de pull request:
   - Um título claro e uma descrição da mudança e do problema que ela resolve
   - Por que você escolheu sua abordagem arquitetônica específica e quais alternativas descartou
   - Os passos exatos que você usou para testar a mudança localmente
   - Referência a quaisquer issues relacionados
   - Screenshots se a mudança afeta a UI
   - Todas as caixas da checklist do modelo marcadas

<!-- prettier-ignore-start -->
:::warning[O modelo é obrigatório]
Pull requests que não preenchem o modelo são fechados automaticamente por um bot. Edite a descrição
do pull request para responder cada seção e marcar cada caixa; o pull request é reaberto
automaticamente.
:::
<!-- prettier-ignore-end -->

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

## Código Assistido por IA

Nós damos boas-vindas ao uso de ferramentas de IA para ajudar a escrever código. No entanto, você é
estritamente responsável pelo código que envia. Você DEVE entendê-lo completamente e ser capaz de
explicar as escolhas arquitetônicas.

Código gerado por IA frequentemente introduz bugs sutis, incompatibilidades arquitetônicas e falhas
de segurança. Pull requests que parecerem saída bruta de IA sem revisão serão fechados para
respeitar o tempo dos mantenedores.

## Diretrizes Adicionais

### Mensagens de Commit

Use mensagens de commit claras e descritivas:

- Comece com letra maiúscula
- Mantenha a primeira linha sob 50 caracteres
- Use o modo imperativo ("Adiciona feature" não "Adicionou feature")
- Inclua mais detalhes no corpo se necessário

### Alterações Pequenas e Focadas

Mantenha pull requests focados em uma única feature ou correção. Alterações grandes devem ser
divididas em partes menores e lógicas.

<!-- prettier-ignore-start -->
:::tip[Discuta Primeiro]
Para mudanças maiores, abra uma
[issue](https://github.com/barebaric/rayforge/issues) primeiro para discutir sua abordagem antes de
investir tempo significativo.
:::
<!-- prettier-ignore-end -->

<!-- prettier-ignore-start -->
:::note[Precisa de Ajuda?]
Se você não tiver certeza sobre qualquer parte do processo de
contribuição, não hesite em pedir ajuda em uma issue ou discussão.
:::
<!-- prettier-ignore-end -->
