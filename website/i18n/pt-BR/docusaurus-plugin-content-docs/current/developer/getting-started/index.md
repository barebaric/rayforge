# Obtendo o Código

Este guia cobre como obter o código fonte do Rayforge para desenvolvimento.

## Faça Fork do Repositório

Faça fork do [repositório Rayforge](https://github.com/barebaric/rayforge) no GitHub para criar sua própria cópia onde você pode fazer alterações.

## Clone Seu Fork

```bash
git clone https://github.com/SEU_USUARIO/rayforge.git
cd rayforge
```

## Adicione o Repositório Upstream

Adicione o repositório original como um remote upstream para acompanhar as alterações:

```bash
git remote add upstream https://github.com/barebaric/rayforge.git
```

## Verifique o Repositório

Verifique se os remotes estão configurados corretamente:

```bash
git remote -v
```

Você deve ver tanto seu fork (origin) quanto o repositório upstream.

## Próximos Passos

Após obter o código, continue com [Configuração](setup) para configurar seu ambiente de desenvolvimento.
