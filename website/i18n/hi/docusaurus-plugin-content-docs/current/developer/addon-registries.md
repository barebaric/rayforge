---
description:
  "Rayforge रजिस्ट्रियों के माध्यम से ऐड-ऑन प्रकाशित और खोजें. अपने एक्सटेंशन लेज़र कटिंग समुदाय के
  साथ साझा करें."
---

# ऐड-ऑन रजिस्ट्रियाँ

रजिस्ट्रियाँ ही हैं जिनके द्वारा Rayforge विस्तारशीलता प्रबंधित करता है. प्रत्येक रजिस्ट्री संबंधित
घटकों का एक संग्रह रखती है — स्टेप, उत्पादक, क्रियाएँ, इत्यादि. जब आपका ऐड-ऑन कुछ पंजीकृत करता है,
तो वह पूरे एप्लिकेशन में उपलब्ध हो जाता है.

## रजिस्ट्रियाँ कैसे काम करती हैं

सभी रजिस्ट्रियाँ एक समान पैटर्न का पालन करती हैं. वे आइटम जोड़ने के लिए एक `register()` विधि और
उन्हें पुनः प्राप्त करने की विभिन्न खोज विधियाँ देती हैं. अधिकांश रजिस्ट्रियाँ यह भी ट्रैक करती हैं
कि प्रत्येक आइटम किस ऐड-ऑन ने पंजीकृत किया, ताकि ऐड-ऑन अनलोड होने पर वे सफ़ाई कर सकें.

यहाँ सामान्य पैटर्न है:

```python
@hookimpl
def register_steps(step_registry):
    from .my_step import MyCustomStep
    step_registry.register(MyCustomStep, addon_name="my_addon")
```

`addon_name` पैरामीटर वैकल्पिक है लेकिन अनुशंसित है. यह सुनिश्चित करता है कि उपयोगकर्ता आपका ऐड-ऑन
अक्षित करने पर आपके घटक ठीक से हटा दिए जाएँ.

## स्टेप रजिस्ट्री

स्टेप रजिस्ट्री (`StepRegistry`) ऑपरेशन पैनल में दिखाई देने वाले स्टेप प्रकार प्रबंधित करती है.
प्रत्येक स्टेप उपयोगकर्ताओं द्वारा अपने जॉब में जोड़े जाने वाले ऑपरेशन के प्रकार को दर्शाता है.

### स्टेप पंजीकृत करना

```python
@hookimpl
def register_steps(step_registry):
    from .my_step import MyCustomStep
    step_registry.register(MyCustomStep, addon_name="my_addon")
```

स्टेप के वर्ग नाम का उपयोग रजिस्ट्री कुंजी के रूप में होता है. आपके स्टेप वर्ग को `Step` से
वंशानुक्रमित होना चाहिए और `TYPELABEL`, `HIDDEN` जैसे गुण परिभाषित करने चाहिए तथा `create()` वर्ग
विधि कार्यान्वित करनी चाहिए.

### स्टेप पुनः प्राप्त करना

रजिस्ट्री स्टेप खोजने के लिए कई विधियाँ देती है:

```python
# Get a step by its class name
step_class = step_registry.get("MyCustomStep")

# Get a step by its TYPELABEL (for backward compatibility)
step_class = step_registry.get_by_typelabel("My Custom Step")

# Get all registered steps
all_steps = step_registry.all_steps()

# Get factory methods for UI menus (excludes hidden steps)
factories = step_registry.get_factories()
```

## उत्पादक रजिस्ट्री

उत्पादक रजिस्ट्री (`ProducerRegistry`) ops उत्पादक प्रबंधित करती है. उत्पादक किसी स्टेप के लिए
टूलपाथ ऑपरेशन जनरेट करते हैं — मूल रूप से, वे आपके वर्कपीस को मशीन निर्देशों में बदलते हैं.

### उत्पादक पंजीकृत करना

```python
@hookimpl
def register_producers(producer_registry):
    from .my_producer import MyCustomProducer
    producer_registry.register(MyCustomProducer, addon_name="my_addon")
```

डिफ़ॉल्ट रूप से, वर्ग नाम रजिस्ट्री कुंजी बन जाता है. आप एक कस्टम नाम निर्दिष्ट कर सकते हैं:

```python
producer_registry.register(MyCustomProducer, name="custom_name", addon_name="my_addon")
```

### उत्पादक पुनः प्राप्त करना

```python
# Get a producer by name
producer_class = producer_registry.get("MyCustomProducer")

# Get all producers
all_producers = producer_registry.all_producers()
```

## ट्रांसफ़ॉर्मर रजिस्ट्री

ट्रांसफ़ॉर्मर रजिस्ट्री (`TransformerRegistry`) ops ट्रांसफ़ॉर्मर प्रबंधित करती है. ट्रांसफ़ॉर्मर
उत्पादकों द्वारा जनरेट होने के बाद ऑपरेशन पोस्ट-प्रोसेस करते हैं — पाथ अनुकूलन, स्मूदिंग, या
होल्डिंग टैब जोड़ने जैसे कार्य सोचें.

### ट्रांसफ़ॉर्मर पंजीकृत करना

```python
@hookimpl
def register_transformers(transformer_registry):
    from .my_transformer import MyCustomTransformer
    transformer_registry.register(MyCustomTransformer, addon_name="my_addon")
```

### ट्रांसफ़ॉर्मर पुनः प्राप्त करना

```python
# Get a transformer by name
transformer_class = transformer_registry.get("MyCustomTransformer")

# Get all transformers
all_transformers = transformer_registry.all_transformers()
```

## क्रिया रजिस्ट्री

क्रिया रजिस्ट्री (`ActionRegistry`) विंडो क्रियाएँ प्रबंधित करती है. क्रियाएँ ही हैं जिनके द्वारा आप
मेनू आइटम, टूलबार बटन, और कीबोर्ड शॉर्टकट जोड़ते हैं. यह अधिक सुविधा-समृद्ध रजिस्ट्रियों में से एक
है.

### क्रिया पंजीकृत करना

```python
from gi.repository import Gio
from rayforge.ui_gtk.action_registry import MenuPlacement, ToolbarPlacement

@hookimpl
def register_actions(action_registry):
    # Create the action
    action = Gio.SimpleAction.new("my-action", None)
    action.connect("activate", lambda a, p: do_something())

    # Register with optional menu and toolbar placement
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

### क्रिया पैरामीटर

क्रिया पंजीकृत करते समय, आप दे सकते हैं:

- `action_name`: क्रिया का पहचानकर्ता ("win." उपसर्ग के बिना)
- `action`: `Gio.SimpleAction` इंस्टेंस
- `addon_name`: सफ़ाई के लिए आपके ऐड-ऑन का नाम
- `label`: मेनू और टूलटिप के लिए सुगम टेक्स्ट
- `icon_name`: टूलबार के लिए आइकन पहचानकर्ता
- `shortcut`: GTK एक्सेलेरेटर सिंटैक्स का उपयोग करता कीबोर्ड शॉर्टकट
- `menu`: `MenuPlacement` ऑब्जेक्ट निर्दिष्ट करते हुए कौन सा मेनू और प्राथमिकता
- `toolbar`: `ToolbarPlacement` ऑब्जेक्ट टूलबार समूह और प्राथमिकता निर्दिष्ट करते हुए

### मेनू स्थापना

`MenuPlacement` वर्ग लेता है:

- `menu_id`: किस मेनू में जोड़ना है (जैसे, "tools", "arrange")
- `priority`: कम संख्याएँ पहले दिखाई देती हैं

### टूलबार स्थापना

`ToolbarPlacement` वर्ग लेता है:

- `group`: टूलबार समूह पहचानकर्ता (जैसे, "main", "arrange")
- `priority`: कम संख्याएँ पहले दिखाई देती हैं

### क्रियाएँ पुनः प्राप्त करना

```python
# Get action info
info = action_registry.get("my-action")

# Get all actions for a specific menu
menu_items = action_registry.get_menu_items("tools")

# Get all actions for a toolbar group
toolbar_items = action_registry.get_toolbar_items("main")

# Get all actions with keyboard shortcuts
shortcuts = action_registry.get_all_with_shortcuts()
```

## कमांड रजिस्ट्री

कमांड रजिस्ट्री (`CommandRegistry`) एडिटर कमांड प्रबंधित करती है. कमांड दस्तावेज़ एडिटर की
कार्यक्षमता विस्तारित करते हैं.

### कमांड पंजीकृत करना

```python
@hookimpl
def register_commands(command_registry):
    from .commands import MyCustomCommand
    command_registry.register("my_command", MyCustomCommand, addon_name="my_addon")
```

कमांड वर्गों को अपने कंस्ट्रक्टर में एक `DocEditor` इंस्टेंस स्वीकार करना चाहिए.

### कमांड पुनः प्राप्त करना

```python
# Get a command by name
command_class = command_registry.get("my_command")

# Get all commands
all_commands = command_registry.all_commands()
```

## एसेट प्रकार रजिस्ट्री

एसेट प्रकार रजिस्ट्री (`AssetTypeRegistry`) उन एसेट प्रकारों को प्रबंधित करती है जो दस्तावेज़ों में
संग्रहीत किए जा सकते हैं. यह गतिशील विक्रमणीकरण सक्षम करता है — जब Rayforge आपके कस्टम एसेट युक्त
दस्तावेज़ लोड करता है, तो उसे पता होता है कि उसे कैसे पुनर्निर्माण करना है.

### एसेट प्रकार पंजीकृत करना

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

`type_name` क्रमबद्ध दस्तावेज़ों में उपयोग होने वाली स्ट्रिंग है जो आपके एसेट प्रकार की पहचान करती
है.

### एसेट प्रकार पुनः प्राप्त करना

```python
# Get an asset class by type name
asset_class = asset_type_registry.get("my_asset")

# Get all registered asset types
all_types = asset_type_registry.all_types()
```

## लेआउट रणनीति रजिस्ट्री

लेआउट रणनीति रजिस्ट्री (`LayoutStrategyRegistry`) दस्तावेज़ एडिटर में सामग्री व्यवस्थित करने के लिए
लेआउट रणनीतियाँ प्रबंधित करती है.

### लेआउट रणनीति पंजीकृत करना

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

ध्यान दें कि लेबल और शॉर्टकट जैसे UI मेटाडेटा क्रिया रजिस्ट्री द्वारा पंजीकृत होने चाहिए, यहाँ नहीं.

### लेआउट रणनीतियाँ पुनः प्राप्त करना

```python
# Get a strategy by name
strategy_class = layout_registry.get("my_layout")

# Get all strategy classes
all_strategies = layout_registry.list_all()

# Get all strategy names
strategy_names = layout_registry.list_names()
```

## आयातक रजिस्ट्री

आयातक रजिस्ट्री (`ImporterRegistry`) फ़ाइल आयातक प्रबंधित करती है. आयातक बाहरी फ़ाइलें Rayforge में
लोड करना संभालते हैं.

### आयातक पंजीकृत करना

```python
@hookimpl
def register_importers(importer_registry):
    from .my_importer import MyCustomImporter
    importer_registry.register(MyCustomImporter, addon_name="my_addon")
```

आपके आयातक वर्ग को `extensions` और `mime_types` वर्ग विशेषताएँ परिभाषित करनी चाहिए ताकि रजिस्ट्री
जाने कि वह कौन सी फ़ाइलें संभालता है.

### आयातक पुनः प्राप्त करना

```python
# Get importer by file extension
importer_class = importer_registry.get_by_extension(".xyz")

# Get importer by MIME type
importer_class = importer_registry.get_by_mime_type("application/x-xyz")

# Get importer by class name
importer_class = importer_registry.get_by_name("MyCustomImporter")

# Get appropriate importer for a file path
importer_class = importer_registry.get_for_file(Path("file.xyz"))

# Get all supported file extensions
extensions = importer_registry.get_supported_extensions()

# Get all file filters for file dialogs
filters = importer_registry.get_all_filters()

# Get importers that support a specific feature
importers = importer_registry.by_feature(ImporterFeature.SOME_FEATURE)
```

## निर्यातक रजिस्ट्री

निर्यातक रजिस्ट्री (`ExporterRegistry`) फ़ाइल निर्यातक प्रबंधित करती है. निर्यातक Rayforge दस्तावेज़
या ऑपरेशन बाहरी फ़ॉर्मेटों में सहेजना संभालते हैं.

### निर्यातक पंजीकृत करना

```python
@hookimpl
def register_exporters(exporter_registry):
    from .my_exporter import MyCustomExporter
    exporter_registry.register(MyCustomExporter, addon_name="my_addon")
```

आपके निर्यातक वर्ग को `extensions` और `mime_types` वर्ग विशेषताएँ परिभाषित करनी चाहिए.

### निर्यातक पुनः प्राप्त करना

```python
# Get exporter by file extension
exporter_class = exporter_registry.get_by_extension(".xyz")

# Get exporter by MIME type
exporter_class = exporter_registry.get_by_mime_type("application/x-xyz")

# Get all file filters for file dialogs
filters = exporter_registry.get_all_filters()
```

## रेंडरर रजिस्ट्री

रेंडरर रजिस्ट्री (`RendererRegistry`) एसेट रेंडरर प्रबंधित करती है. रेंडरर एसेट UI में दिखाते हैं.

### रेंडरर पंजीकृत करना

```python
@hookimpl
def register_renderers(renderer_registry):
    from .my_renderer import MyAssetRenderer
    renderer_registry.register(MyAssetRenderer(), addon_name="my_addon")
```

ध्यान दें कि आप रेंडरर इंस्टेंस पंजीकृत करते हैं, वर्ग नहीं. रेंडरर के वर्ग नाम का उपयोग रजिस्ट्री
कुंजी के रूप में होता है.

### रेंडरर पुनः प्राप्त करना

```python
# Get renderer by class name
renderer = renderer_registry.get("MyAssetRenderer")

# Get renderer by name (same as get)
renderer = renderer_registry.get_by_name("MyAssetRenderer")

# Get all renderers
all_renderers = renderer_registry.all()
```

## लाइब्रेरी मैनेजर

लाइब्रेरी मैनेजर (`LibraryManager`) सामग्री लाइब्रेरी प्रबंधित करता है. हालाँकि तकनीकी रूप से
रजिस्ट्री नहीं, वह ऐड-ऑन-द्वारा-दी गई लाइब्रेरी पंजीकृत करने के लिए समान पैटर्न का पालन करता है.

### सामग्री लाइब्रेरी पंजीकृत करना

```python
@hookimpl
def register_material_libraries(library_manager):
    from pathlib import Path
    lib_path = Path(__file__).parent / "materials"
    library_manager.add_library_from_path(lib_path)
```

पंजीकृत लाइब्रेरी डिफ़ॉल्ट रूप से केवल-पढ़ने योग्य होती हैं. उपयोगकर्ता सामग्रियाँ देख और उपयोग कर
सकते हैं लेकिन उन्हें UI द्वारा संशोधित नहीं कर सकते.
