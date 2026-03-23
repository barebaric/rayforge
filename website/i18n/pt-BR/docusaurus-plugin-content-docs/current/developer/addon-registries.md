# Registros de Addons

Registros são como o Rayforge gerencia a extensibilidade. Cada registro mantém uma coleção de componentes relacionados—passos, produtores, ações e assim por diante. Quando seu addon registra algo, ele se torna disponível em toda a aplicação.

## Como os Registros Funcionam

Todos os registros seguem um padrão similar. Eles fornecem um método `register()` para adicionar itens, e vários métodos de busca para recuperá-los. A maioria dos registros também rastreia qual addon registrou cada item, para que possam limpar quando um addon é descarregado.

Aqui está o padrão geral:

```python
@hookimpl
def register_steps(step_registry):
    from .my_step import MyCustomStep
    step_registry.register(MyCustomStep, addon_name="my_addon")
```

O parâmetro `addon_name` é opcional mas recomendado. Ele garante que seus componentes sejam removidos corretamente se o usuário desabilitar seu addon.

## Registro de Passos

O registro de passos (`StepRegistry`) gerencia tipos de passos que aparecem no painel de operações. Cada passo representa um tipo de operação que os usuários podem adicionar ao seu trabalho.

### Registrando um Passo

```python
@hookimpl
def register_steps(step_registry):
    from .my_step import MyCustomStep
    step_registry.register(MyCustomStep, addon_name="my_addon")
```

O nome da classe do passo é usado como chave do registro. Sua classe de passo deve herdar de `Step` e definir atributos como `TYPELABEL`, `HIDDEN`, e implementar o método de classe `create()`.

### Recuperando Passos

O registro fornece vários métodos para buscar passos:

```python
# Obtém um passo pelo nome de sua classe
step_class = step_registry.get("MyCustomStep")

# Obtém um passo pelo seu TYPELABEL (para compatibilidade retroativa)
step_class = step_registry.get_by_typelabel("My Custom Step")

# Obtém todos os passos registrados
all_steps = step_registry.all_steps()

# Obtém métodos fábrica para menus de interface (exclui passos ocultos)
factories = step_registry.get_factories()
```

## Registro de Produtores

O registro de produtores (`ProducerRegistry`) gerencia produtores de ops. Produtores geram as operações de trajetória para um passo—essencialmente, eles convertem sua peça em instruções de máquina.

### Registrando um Produtor

```python
@hookimpl
def register_producers(producer_registry):
    from .my_producer import MyCustomProducer
    producer_registry.register(MyCustomProducer, addon_name="my_addon")
```

Por padrão, o nome da classe se torna a chave do registro. Você pode especificar um nome personalizado:

```python
producer_registry.register(MyCustomProducer, name="custom_name", addon_name="my_addon")
```

### Recuperando Produtores

```python
# Obtém um produtor pelo nome
producer_class = producer_registry.get("MyCustomProducer")

# Obtém todos os produtores
all_producers = producer_registry.all_producers()
```

## Registro de Transformadores

O registro de transformadores (`TransformerRegistry`) gerencia transformadores de ops. Transformadores fazem pós-processamento das operações depois que os produtores as geram—pense em tarefas como otimização de caminho, suavização ou adição de abas de fixação.

### Registrando um Transformador

```python
@hookimpl
def register_transformers(transformer_registry):
    from .my_transformer import MyCustomTransformer
    transformer_registry.register(MyCustomTransformer, addon_name="my_addon")
```

### Recuperando Transformadores

```python
# Obtém um transformador pelo nome
transformer_class = transformer_registry.get("MyCustomTransformer")

# Obtém todos os transformadores
all_transformers = transformer_registry.all_transformers()
```

## Registro de Ações

O registro de ações (`ActionRegistry`) gerencia ações de janela. Ações são como você adiciona itens de menu, botões de barra de ferramentas e atalhos de teclado. Este é um dos registros mais ricos em recursos.

### Registrando uma Ação

```python
from gi.repository import Gio
from rayforge.ui_gtk.action_registry import MenuPlacement, ToolbarPlacement

@hookimpl
def register_actions(action_registry):
    # Cria a ação
    action = Gio.SimpleAction.new("my-action", None)
    action.connect("activate", lambda a, p: do_something())
    
    # Registra com posicionamento opcional em menu e barra de ferramentas
    action_registry.register(
        action_name="my-action",
        action=action,
        addon_name="my_addon",
        label="My Action",
        icon_name="document-new-symbolic",
        shortcut="<Ctrl><Alt>m",
        menu=MenuPlacement(menu_id="tools", priority=50),
        toolbar=ToolbarPlacement(group="main", priority=50),
    )
```

### Parâmetros de Ação

Ao registrar uma ação, você pode fornecer:

- `action_name`: O identificador da ação (sem o prefixo "win.")
- `action`: A instância de `Gio.SimpleAction`
- `addon_name`: O nome do seu addon para limpeza
- `label`: Texto legível para humanos para menus e dicas
- `icon_name`: Identificador de ícone para barras de ferramentas
- `shortcut`: Atalho de teclado usando sintaxe de acelerador GTK
- `menu`: Objeto `MenuPlacement` especificando qual menu e prioridade
- `toolbar`: Objeto `ToolbarPlacement` especificando grupo da barra de ferramentas e prioridade

### Posicionamento em Menu

A classe `MenuPlacement` aceita:

- `menu_id`: Em qual menu adicionar (ex: "tools", "arrange")
- `priority`: Números menores aparecem primeiro

### Posicionamento em Barra de Ferramentas

A classe `ToolbarPlacement` aceita:

- `group`: Identificador do grupo da barra de ferramentas (ex: "main", "arrange")
- `priority`: Números menores aparecem primeiro

### Recuperando Ações

```python
# Obtém informações da ação
info = action_registry.get("my-action")

# Obtém todas as ações para um menu específico
menu_items = action_registry.get_menu_items("tools")

# Obtém todas as ações para um grupo da barra de ferramentas
toolbar_items = action_registry.get_toolbar_items("main")

# Obtém todas as ações com atalhos de teclado
shortcuts = action_registry.get_all_with_shortcuts()
```

## Registro de Comandos

O registro de comandos (`CommandRegistry`) gerencia comandos do editor. Comandos estendem a funcionalidade do editor de documentos.

### Registrando um Comando

```python
@hookimpl
def register_commands(command_registry):
    from .commands import MyCustomCommand
    command_registry.register("my_command", MyCustomCommand, addon_name="my_addon")
```

Classes de comando devem aceitar uma instância de `DocEditor` em seu construtor.

### Recuperando Comandos

```python
# Obtém um comando pelo nome
command_class = command_registry.get("my_command")

# Obtém todos os comandos
all_commands = command_registry.all_commands()
```

## Registro de Tipos de Asset

O registro de tipos de asset (`AssetTypeRegistry`) gerencia tipos de assets que podem ser armazenados em documentos. Isso habilita desserialização dinâmica—quando o Rayforge carrega um documento contendo seu asset personalizado, ele sabe como reconstruí-lo.

### Registrando um Tipo de Asset

```python
@hookimpl
def register_asset_types(asset_type_registry):
    from .my_asset import MyCustomAsset
    asset_type_registry.register(
        MyCustomAsset,
        type_name="my_asset",
        addon_name="my_addon"
    )
```

O `type_name` é a string usada em documentos serializados para identificar seu tipo de asset.

### Recuperando Tipos de Asset

```python
# Obtém uma classe de asset pelo nome do tipo
asset_class = asset_type_registry.get("my_asset")

# Obtém todos os tipos de assets registrados
all_types = asset_type_registry.all_types()
```

## Registro de Estratégias de Layout

O registro de estratégias de layout (`LayoutStrategyRegistry`) gerencia estratégias de layout para organizar conteúdo no editor de documentos.

### Registrando uma Estratégia de Layout

```python
@hookimpl
def register_layout_strategies(layout_registry):
    from .my_layout import MyLayoutStrategy
    layout_registry.register(
        MyLayoutStrategy,
        name="my_layout",
        addon_name="my_addon"
    )
```

Note que metadados de interface como rótulos e atalhos devem ser registrados via registro de ações, não aqui.

### Recuperando Estratégias de Layout

```python
# Obtém uma estratégia pelo nome
strategy_class = layout_registry.get("my_layout")

# Obtém todas as classes de estratégia
all_strategies = layout_registry.list_all()

# Obtém todos os nomes de estratégia
strategy_names = layout_registry.list_names()
```

## Registro de Importadores

O registro de importadores (`ImporterRegistry`) gerencia importadores de arquivos. Importadores lidam com o carregamento de arquivos externos no Rayforge.

### Registrando um Importador

```python
@hookimpl
def register_importers(importer_registry):
    from .my_importer import MyCustomImporter
    importer_registry.register(MyCustomImporter, addon_name="my_addon")
```

Sua classe de importador deve definir atributos de classe `extensions` e `mime_types` para que o registro saiba quais arquivos ele manipula.

### Recuperando Importadores

```python
# Obtém importador por extensão de arquivo
importer_class = importer_registry.get_by_extension(".xyz")

# Obtém importador por tipo MIME
importer_class = importer_registry.get_by_mime_type("application/x-xyz")

# Obtém importador pelo nome da classe
importer_class = importer_registry.get_by_name("MyCustomImporter")

# Obtém importador apropriado para um caminho de arquivo
importer_class = importer_registry.get_for_file(Path("file.xyz"))

# Obtém todas as extensões de arquivo suportadas
extensions = importer_registry.get_supported_extensions()

# Obtém todos os filtros de arquivo para diálogos de arquivo
filters = importer_registry.get_all_filters()

# Obtém importadores que suportam um recurso específico
importers = importer_registry.by_feature(ImporterFeature.SOME_FEATURE)
```

## Registro de Exportadores

O registro de exportadores (`ExporterRegistry`) gerencia exportadores de arquivos. Exportadores lidam com o salvamento de documentos ou operações do Rayforge para formatos externos.

### Registrando um Exportador

```python
@hookimpl
def register_exporters(exporter_registry):
    from .my_exporter import MyCustomExporter
    exporter_registry.register(MyCustomExporter, addon_name="my_addon")
```

Sua classe de exportador deve definir atributos de classe `extensions` e `mime_types`.

### Recuperando Exportadores

```python
# Obtém exportador por extensão de arquivo
exporter_class = exporter_registry.get_by_extension(".xyz")

# Obtém exportador por tipo MIME
exporter_class = exporter_registry.get_by_mime_type("application/x-xyz")

# Obtém todos os filtros de arquivo para diálogos de arquivo
filters = exporter_registry.get_all_filters()
```

## Registro de Renderizadores

O registro de renderizadores (`RendererRegistry`) gerencia renderizadores de assets. Renderizadores exibem assets na interface.

### Registrando um Renderizador

```python
@hookimpl
def register_renderers(renderer_registry):
    from .my_renderer import MyAssetRenderer
    renderer_registry.register(MyAssetRenderer(), addon_name="my_addon")
```

Note que você registra uma instância de renderizador, não uma classe. O nome da classe do renderizador é usado como chave do registro.

### Recuperando Renderizadores

```python
# Obtém renderizador pelo nome da classe
renderer = renderer_registry.get("MyAssetRenderer")

# Obtém renderizador pelo nome (mesmo que get)
renderer = renderer_registry.get_by_name("MyAssetRenderer")

# Obtém todos os renderizadores
all_renderers = renderer_registry.all()
```

## Gerenciador de Bibliotecas

O gerenciador de bibliotecas (`LibraryManager`) gerencia bibliotecas de materiais. Embora não seja tecnicamente um registro, segue padrões similares para registrar bibliotecas fornecidas por addons.

### Registrando uma Biblioteca de Materiais

```python
@hookimpl
def register_material_libraries(library_manager):
    from pathlib import Path
    lib_path = Path(__file__).parent / "materials"
    library_manager.add_library_from_path(lib_path)
```

Bibliotecas registradas são somente leitura por padrão. Usuários podem visualizar e usar os materiais, mas não podem modificá-los através da interface.
