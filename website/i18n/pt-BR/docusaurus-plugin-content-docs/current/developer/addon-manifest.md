# Manifesto do Addon

Todo addon precisa de um arquivo `rayforge-addon.yaml` em seu diretório raiz. Este manifesto informa ao Rayforge sobre seu addon—seu nome, o que ele fornece e como carregá-lo.

## Estrutura Básica

Aqui está um manifesto completo com todos os campos comuns:

```yaml
name: my_custom_addon
display_name: "My Custom Addon"
description: "Adds support for the XYZ laser cutter."
api_version: 9
url: https://github.com/username/my-custom-addon

author:
  name: Jane Doe
  email: jane@example.com

depends:
  - rayforge>=0.27.0

requires:
  - some-other-addon>=1.0.0

provides:
  backend: my_addon.backend
  frontend: my_addon.frontend
  assets:
    - path: assets/profiles.json
      type: profiles

license:
  name: MIT
```

## Campos Obrigatórios

### `name`

Um identificador único para seu addon. Deve ser um nome de módulo Python válido—apenas letras, números e underscores, e não pode começar com um número.

```yaml
name: my_custom_addon
```

### `display_name`

Um nome legível para humanos exibido na interface. Pode conter espaços e caracteres especiais.

```yaml
display_name: "My Custom Addon"
```

### `description`

Uma breve descrição do que seu addon faz. Aparece no gerenciador de addons.

```yaml
description: "Adds support for the XYZ laser cutter."
```

### `api_version`

A versão da API que seu addon destina. Deve ser pelo menos 1 (a versão mínima suportada) e no máximo a versão atual (9). Usar uma versão maior que a suportada fará seu addon falhar na validação.

```yaml
api_version: 9
```

Consulte a documentação de [Hooks](./addon-hooks.md#api-version-history) para o que mudou em cada versão.

### `author`

Informações sobre o autor do addon. O campo `name` é obrigatório; `email` é opcional mas recomendado para os usuários entrarem em contato com você.

```yaml
author:
  name: Jane Doe
  email: jane@example.com
```

## Campos Opcionais

### `url`

Uma URL para a página inicial ou repositório do seu addon.

```yaml
url: https://github.com/username/my-custom-addon
```

### `depends`

Restrições de versão para o próprio Rayforge. Especifique a versão mínima que seu addon requer.

```yaml
depends:
  - rayforge>=0.27.0
```

### `requires`

Dependências de outros addons. Liste nomes de addons com restrições de versão.

```yaml
requires:
  - some-other-addon>=1.0.0
```

### `version`

O número de versão do seu addon. Isso é tipicamente determinado automaticamente a partir de tags git, mas você pode especificá-lo explicitamente. Use versionamento semântico (ex: `1.0.0`).

```yaml
version: 1.0.0
```

## Pontos de Entrada

A seção `provides` define o que seu addon contribui para o Rayforge.

### Backend

O módulo backend é carregado tanto no processo principal quanto nos processos de trabalho. Use-o para drivers de máquinas, tipos de passos, produtores de ops e qualquer funcionalidade principal.

```yaml
provides:
  backend: my_addon.backend
```

O valor é um caminho de módulo Python pontilhado relativo ao diretório do seu addon.

### Frontend

O módulo frontend é carregado apenas no processo principal. Use-o para componentes de interface, widgets GTK e qualquer coisa que precise da janela principal.

```yaml
provides:
  frontend: my_addon.frontend
```

### Assets

Você pode empacotar arquivos de assets que o Rayforge reconhecerá. Cada asset tem um caminho e tipo:

```yaml
provides:
  assets:
    - path: assets/profiles.json
      type: profiles
    - path: assets/templates
      type: templates
```

O `path` é relativo à raiz do seu addon e deve existir. Os tipos de assets são definidos pelo Rayforge e podem incluir coisas como perfis de máquinas, bibliotecas de materiais ou templates.

## Informações de Licença

O campo `license` descreve como seu addon é licenciado. Para addons gratuitos, basta especificar o nome da licença usando um identificador SPDX:

```yaml
license:
  name: MIT
```

Identificadores SPDX comuns incluem `MIT`, `Apache-2.0`, `GPL-3.0` e `BSD-3-Clause`.

## Addons Pagos

O Rayforge suporta addons pagos através da validação de licenças do Gumroad. Se você quiser vender seu addon, pode configurá-lo para exigir uma licença válida antes de funcionar.

### Configuração Básica para Addon Pago

```yaml
license:
  name: BSL-1.1
  required: true
  purchase_url: https://gum.co/my-addon
```

Quando `required` é true, o Rayforge verificará se há uma licença válida antes de carregar seu addon. A `purchase_url` é mostrada aos usuários que não têm uma licença.

### ID do Produto Gumroad

Adicione o ID do seu produto Gumroad para habilitar a validação de licença:

```yaml
license:
  name: BSL-1.1
  required: true
  purchase_url: https://gum.co/my-addon
  product_id: "abc123def456"
```

Para múltiplos IDs de produtos (ex: diferentes níveis de preço):

```yaml
license:
  name: BSL-1.1
  required: true
  purchase_url: https://gum.co/my-addon
  product_ids:
    - "abc123def456"
    - "xyz789ghi012"
```

### Exemplo Completo de Addon Pago

Aqui está um manifesto completo para um addon pago:

```yaml
name: premium_laser_pack
display_name: "Premium Laser Pack"
description: "Advanced features for professional laser cutting."
api_version: 9
url: https://example.com/premium-laser-pack

author:
  name: Your Name
  email: you@example.com

depends:
  - rayforge>=0.27.0

provides:
  backend: premium_pack.backend
  frontend: premium_pack.frontend

license:
  name: BSL-1.1
  required: true
  purchase_url: https://gum.co/premium-laser-pack
  product_ids:
    - "standard_tier_id"
    - "pro_tier_id"
```

### Verificando o Status da Licença no Código

No código do seu addon, você pode verificar se uma licença é válida:

```python
@hookimpl
def rayforge_init(context):
    if context.license_validator:
        # Verifica se o usuário tem uma licença válida para seu produto
        is_valid = context.license_validator.is_product_valid("your_product_id")
        if not is_valid:
            # Opcionalmente mostra uma mensagem ou limita a funcionalidade
            logger.warning("License not found - some features disabled")
```

## Regras de Validação

O Rayforge valida seu manifesto ao carregar o addon. Aqui estão as regras:

O `name` deve ser um identificador Python válido (letras, números, underscores, sem números no início). O `api_version` deve ser um inteiro entre 1 e a versão atual. O `author.name` não pode estar vazio ou conter texto placeholder como "your-github-username". Pontos de entrada devem ser caminhos de módulo válidos e os módulos devem existir. Caminhos de assets devem ser relativos (sem `..` ou `/` no início) e os arquivos devem existir.

Se a validação falhar, o Rayforge registra um erro e pula seu addon. Verifique a saída do console durante o desenvolvimento para capturar esses problemas.
