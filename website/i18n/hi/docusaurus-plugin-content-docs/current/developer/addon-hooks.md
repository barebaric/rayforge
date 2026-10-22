---
description:
  "Rayforge में ऐड-ऑन हुक - लेज़र कटिंग वर्कफ़्लो में कस्टम कार्यक्षमता एकीकृत करने के लिए जीवनचक्र
  घटनाएँ और विस्तार बिंदु."
---

# ऐड-ऑन हुक

हुक आपके ऐड-ऑन और Rayforge के बीच के कनेक्शन बिंदु हैं. जब एप्लिकेशन में कुछ होता है — एक स्टेप बनता
है, एक संवाद खुलता है, या विंडो आरंभीकृत होती है — तो Rayforge कोई भी पंजीकृत हुक कॉल करता है ताकि
आपका ऐड-ऑन प्रतिक्रिया दे सके.

## हुक कैसे काम करते हैं

Rayforge अपनी हुक प्रणाली के लिए [pluggy](https://pluggy.readthedocs.io/) उपयोग करता है. हुक
कार्यान्वित करने के लिए, एक फ़ंक्शन को `@pluggy.HookimplMarker("rayforge")` से सजाएँ:

```python
import pluggy

hookimpl = pluggy.HookimplMarker("rayforge")

@hookimpl
def rayforge_init(context):
    # Your code runs when Rayforge finishes initializing
    pass
```

आपको हर हुक कार्यान्वित करने की आवश्यकता नहीं — केवल उन्हीं जिनकी आपको आवश्यकता है. सभी हुक वैकल्पिक
हैं.

## जीवनचक्र हुक

ये हुक आपके ऐड-ऑन के समग्र जीवनचक्र संभालते हैं.

### `rayforge_init(context)`

यह आपका मुख्य एंट्री पॉइंट है. Rayforge यह हुक एप्लिकेशन संदर्भ पूरी तरह आरंभीकृत होने के बाद कॉल
करता है, यानी सभी मैनेजर, कॉन्फ़िग, और हार्डवेयर तैयार हैं. सामान्य सेटअप, लॉगिंग, या UI तत्व जोड़ने
के लिए इसका उपयोग करें.

`context` पैरामीटर एक `RayforgeContext` इंस्टेंस है जो आपको Rayforge की हर चीज़ तक पहुँच देता है.
विवरण के लिए [Rayforge डेटा तक पहुँच](./addon-overview.md#accessing-rayforges-data) देखें.

```python
@hookimpl
def rayforge_init(context):
    logger.info("My addon is starting up!")
    machine = context.machine
    if machine:
        logger.info(f"Running on machine: {machine.id}")
```

### `on_unload()`

Rayforge यह कॉल करता है जब आपका ऐड-ऑन अक्षित या अनलोड किया जा रहा हो. संसाधन साफ़ करने, कनेक्शन बंद
करने, या हैंडलर अपंजीकृत करने के लिए इसका उपयोग करें.

```python
@hookimpl
def on_unload():
    logger.info("My addon is shutting down")
    # Clean up any resources here
```

### `main_window_ready(main_window)`

यह हुक मुख्य विंडो पूरी तरह आरंभीकृत होने पर चलता है. UI पृष्ठ, कमांड, या अन्य घटक पंजीकृत करने के
लिए उपयोगी है जिन्हें पहले मुख्य विंडो का मौजूद होना चाहिए.

`main_window` पैरामीटर `MainWindow` इंस्टेंस है.

```python
@hookimpl
def main_window_ready(main_window):
    # Add a custom page to the main window
    from .my_page import MyCustomPage
    main_window.add_page("my-page", MyCustomPage())
```

## जॉब मॉनिटरिंग हुक

ये हुक मशीन पर चल रहे जॉब के जीवनचक्र और प्रगति की जानकारी देते हैं, ताकि ऐड-ऑन तब प्रतिक्रिया दे
सकें जब जॉब शुरू हो, आगे बढ़े, रुके, फिर शुरू हो या पूरा हो — उदाहरण के लिए, बाहरी कमांड चलाना या
स्मार्ट-होम सिस्टम अपडेट करना.

सभी को एक `JobStatus` स्नैपशॉट (`rayforge.machine.job_status.JobStatus`) मिलता है, जो इन फ़ील्ड वाला
अपरिवर्तनीय ऑब्जेक्ट है:

- `machine_id` और `machine_name`: जॉब चलाने वाली मशीन की पहचान
- `kind`: बर्न जॉब के लिए `JobKind.SEND`, आउटलाइन ट्रेस के लिए `JobKind.FRAME`
- `phase`: `JobPhase.RUNNING` या `JobPhase.PAUSED`
- `started_at`: जॉब शुरू होने का वॉल-क्लॉक समय (युग से सेकंड)
- `elapsed_seconds`: जॉब शुरू होने के बाद बीते सेकंड
- `total_distance` और `traveled_distance`: पथ की कुल और तय की गई लंबाई, मिलीमीटर में
- `progress_fraction`: जॉब का पूरा हुआ अंश, `0.0` से `1.0` तक
- `eta_seconds`: अनुमानित शेष सेकंड, या अज्ञात होने पर `None`
- `estimated_seconds`: जॉब की मॉडल की गई कुल अवधि सेकंड में, या अज्ञात होने पर `None`

हैंडलर मुख्य थ्रेड पर चलते हैं और जल्दी लौटने चाहिए; लंबा काम (जैसे सबप्रोसेस शुरू करना या नेटवर्क
कॉल) बैकग्राउंड थ्रेड को सौंपें.

### `on_job_started(status)`

जब कोई जॉब किसी मशीन पर चलना शुरू होता है तब बुलाया जाता है.

```python
@hookimpl
def on_job_started(status):
    logger.info(f"Job on {status.machine_name} started")
```

### `on_job_updated(status)`

जॉब चलने के दौरान प्रगति अपडेट के साथ बुलाया जाता है, अधिकतम लगभग एक बार प्रति सेकंड.

```python
@hookimpl
def on_job_updated(status):
    print(
        f"Progress: {status.progress_fraction:.0%}, "
        f"ETA {status.eta_seconds}s"
    )
```

### `on_job_paused(status)` / `on_job_resumed(status)`

तब बुलाया जाता है जब चल रहा जॉब फीड होल्ड (रुकना) में जाता है या उससे बाहर निकलता है.

### `on_job_finished(status, outcome, message="")`

जब कोई जॉब पूरा होता है, चाहे किसी भी तरह हुआ हो, तब बुलाया जाता है. `outcome` तर्क `JobOutcome` का
सदस्य है — `COMPLETED`, `CANCELLED` या `FAILED` — और `message` में विफलता पर त्रुटि विवरण होता है.
पूरे हुए जॉब के लिए स्नैपशॉट का `progress_fraction` `1.0` होता है.

```python
@hookimpl
def on_job_finished(status, outcome, message=""):
    if outcome == JobOutcome.FAILED:
        notify(f"Job failed: {message}")
    else:
        notify(f"Job {outcome.name.lower()} on {status.machine_name}")
```

नवीनतम स्नैपशॉट घटनाओं के बीच भी पूछा जा सकता है: जॉब चलते समय `machine.current_job` उसका
`JobStatus` लौटाता है, और कोई जॉब न चल रहा हो तो `None`.

यहाँ एक पूर्ण ऐड-ऑन है जो जॉब की हर स्थिति परिवर्तन पर एक शेल कमांड चलाता है:

```python
import subprocess

import pluggy

from rayforge.machine.job_status import JobOutcome

hookimpl = pluggy.HookimplMarker("rayforge")


@hookimpl
def on_job_started(status):
    subprocess.Popen(["/usr/local/bin/notify-ha", "started"])


@hookimpl
def on_job_paused(status):
    subprocess.Popen(["/usr/local/bin/notify-ha", "paused"])


@hookimpl
def on_job_resumed(status):
    subprocess.Popen(["/usr/local/bin/notify-ha", "resumed"])


@hookimpl
def on_job_finished(status, outcome, message=""):
    subprocess.Popen(["/usr/local/bin/notify-ha", outcome.name.lower()])
```

## पंजीकरण हुक

ये हुक आपको Rayforge की विभिन्न रजिस्ट्रियों के साथ कस्टम घटक पंजीकृत करने देते हैं.

### `register_machines(machine_manager)`

नए मशीन ड्राइवर पंजीकृत करने के लिए इसका उपयोग करें. `machine_manager` एक `MachineManager` इंस्टेंस
है जो सभी मशीन कॉन्फ़िगरेशन प्रबंधित करता है.

```python
@hookimpl
def register_machines(machine_manager):
    from .my_driver import MyCustomMachine
    machine_manager.register("my_custom_machine", MyCustomMachine)
```

### `register_steps(step_registry)`

ऑपरेशन पैनल में दिखाई देने वाले कस्टम स्टेप प्रकार पंजीकृत करें. `step_registry` एक `StepRegistry`
इंस्टेंस है.

```python
@hookimpl
def register_steps(step_registry):
    from .my_step import MyCustomStep
    step_registry.register(MyCustomStep)
```

### `register_producers(producer_registry)`

टूलपाथ जनरेट करने वाले कस्टम ops उत्पादक पंजीकृत करें. `producer_registry` एक `ProducerRegistry`
इंस्टेंस है.

```python
@hookimpl
def register_producers(producer_registry):
    from .my_producer import MyProducer
    producer_registry.register(MyProducer)
```

### `register_transformers(transformer_registry)`

पोस्ट-प्रोसेसिंग ऑपरेशनों के लिए कस्टम ops ट्रांसफ़ॉर्मर पंजीकृत करें. ट्रांसफ़ॉर्मर उत्पादकों
द्वारा जनरेट होने के बाद ऑपरेशन संशोधित करते हैं. `transformer_registry` एक `TransformerRegistry`
इंस्टेंस है.

```python
@hookimpl
def register_transformers(transformer_registry):
    from .my_transformer import MyTransformer
    transformer_registry.register(MyTransformer)
```

### `register_commands(command_registry)`

दस्तावेज़ एडिटर की कार्यक्षमता विस्तारित करने वाले एडिटर कमांड पंजीकृत करें. `command_registry` एक
`CommandRegistry` इंस्टेंस है.

```python
@hookimpl
def register_commands(command_registry):
    from .commands import MyCustomCommand
    command_registry.register("my_command", MyCustomCommand)
```

### `register_actions(action_registry)`

वैकल्पिक मेनू और टूलबार स्थापना के साथ विंडो क्रियाएँ पंजीकृत करें. क्रियाएँ ही हैं जिनके द्वारा आप
बटन, मेनू आइटम, और कीबोर्ड शॉर्टकट जोड़ते हैं. `action_registry` एक `ActionRegistry` इंस्टेंस है.

```python
from gi.repository import Gio
from rayforge.ui_gtk.action_registry import MenuPlacement, ToolbarPlacement

@hookimpl
def register_actions(action_registry):
    action = Gio.SimpleAction.new("my-action", None)
    action.connect("activate", on_my_action_activated)

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

### `register_layout_strategies(layout_registry)`

दस्तावेज़ में सामग्री व्यवस्थित करने के लिए कस्टम लेआउट रणनीतियाँ पंजीकृत करें. `layout_registry` एक
`LayoutStrategyRegistry` इंस्टेंस है. ध्यान दें कि लेबल और शॉर्टकट जैसे UI मेटाडेटा यहाँ नहीं,
`register_actions` द्वारा पंजीकृत होने चाहिए.

```python
@hookimpl
def register_layout_strategies(layout_registry):
    from .my_layout import MyLayoutStrategy
    layout_registry.register(MyLayoutStrategy, name="my_layout")
```

### `register_asset_types(asset_type_registry)`

ऐसे कस्टम एसेट प्रकार पंजीकृत करें जो दस्तावेज़ों में संग्रहीत किए जा सकें. यह ऐड-ऑन-द्वारा-दिए गए
एसेटों का गतिशील विक्रमणीकरण सक्षम करता है. `asset_type_registry` एक `AssetTypeRegistry` इंस्टेंस
है.

```python
@hookimpl
def register_asset_types(asset_type_registry):
    from .my_asset import MyCustomAsset
    asset_type_registry.register(MyCustomAsset, type_name="my_asset")
```

### `register_renderers(renderer_registry)`

अपने एसेट प्रकार UI में दिखाने के लिए कस्टम रेंडरर पंजीकृत करें. `renderer_registry` एक
`RendererRegistry` इंस्टेंस है.

```python
@hookimpl
def register_renderers(renderer_registry):
    from .my_renderer import MyAssetRenderer
    renderer_registry.register(MyAssetRenderer())
```

### `register_exporters(exporter_registry)`

कस्टम निर्यात फ़ॉर्मेटों के लिए फ़ाइल निर्यातक पंजीकृत करें. `exporter_registry` एक
`ExporterRegistry` इंस्टेंस है.

```python
@hookimpl
def register_exporters(exporter_registry):
    from .my_exporter import MyCustomExporter
    exporter_registry.register(MyCustomExporter)
```

### `register_importers(importer_registry)`

कस्टम आयात फ़ॉर्मेटों के लिए फ़ाइल आयातक पंजीकृत करें. `importer_registry` एक `ImporterRegistry`
इंस्टेंस है.

```python
@hookimpl
def register_importers(importer_registry):
    from .my_importer import MyCustomImporter
    importer_registry.register(MyCustomImporter)
```

### `register_material_libraries(library_manager)`

अतिरिक्त सामग्री लाइब्रेरी पंजीकृत करें. सामग्री YAML फ़ाइलें युक्त निर्देशिकाएँ पंजीकृत करने के लिए
`library_manager.add_library_from_path(path)` कॉल करें. डिफ़ॉल्ट रूप से, पंजीकृत लाइब्रेरी
केवल-पढ़ने योग्य होती हैं.

```python
@hookimpl
def register_material_libraries(library_manager):
    from pathlib import Path
    lib_path = Path(__file__).parent / "materials"
    library_manager.add_library_from_path(lib_path)
```

## UI विस्तार हुक

ये हुक आपको मौजूदा UI घटक विस्तारित करने देते हैं.

### `step_settings_loaded(dialog, step, producer)`

Rayforge यह कॉल करता है जब एक स्टेप सेटिंग्स संवाद भरा जा रहा हो. आप स्टेप के उत्पादक प्रकार के आधार
पर संवाद में कस्टम विजेट जोड़ सकते हैं.

`dialog` एक `GeneralStepSettingsView` इंस्टेंस है. `step` कॉन्फ़िगर हो रहा `Step` है. `producer`
`OpsProducer` इंस्टेंस है, या उपलब्ध न होने पर `None`.

```python
@hookimpl
def step_settings_loaded(dialog, step, producer):
    # Only add widgets for specific producer types
    if producer and producer.__class__.__name__ == "MyCustomProducer":
        from .my_widget import create_custom_widget
        dialog.add_widget(create_custom_widget(step))
```

### `transformer_settings_loaded(dialog, step, transformer)`

पोस्ट-प्रोसेसिंग सेटिंग्स भरे जाने पर कॉल होता है. अपने ट्रांसफ़ॉर्मरों के लिए कस्टम विजेट यहाँ
जोड़ें.

`dialog` एक `PostProcessingSettingsView` इंस्टेंस है. `step` कॉन्फ़िगर हो रहा `Step` है.
`transformer` `OpsTransformer` इंस्टेंस है.

```python
@hookimpl
def transformer_settings_loaded(dialog, step, transformer):
    if transformer.__class__.__name__ == "MyCustomTransformer":
        from .my_widget import create_transformer_widget
        dialog.add_widget(create_transformer_widget(transformer))
```

## API संस्करण इतिहास {#api-version-history}

पश्च-संगतता बनाए रखने के लिए हुक संस्करणित होते हैं. नए हुक जोड़ने या मौजूदा बदलने पर API संस्करण
बढ़ाया जाता है. आपके ऐड-ऑन का `api_version` फ़ील्ड कम से कम न्यूनतम समर्थित संस्करण होना चाहिए.

वर्तमान API संस्करण 24 है. हाल के संस्करणों में क्या बदला:

**संस्करण 24** ने जॉब मॉनिटरिंग हुक `on_job_started`, `on_job_updated`, `on_job_paused`,
`on_job_resumed`, और `on_job_finished`, साथ ही `Machine.current_job` पोलिंग API जोड़ी.

**संस्करण 9** ने `main_window_ready`, `register_exporters`, `register_importers`, और
`register_renderers` जोड़े.

**संस्करण 8** ने कस्टम एसेट प्रकारों के लिए `register_asset_types` जोड़ा.

**संस्करण 7** ने `register_material_libraries` जोड़ा.

**संस्करण 6** ने `register_transformers` जोड़ा.

**संस्करण 5** ने `register_step_widgets` को `step_settings_loaded` और `transformer_settings_loaded`
से बदला.

**संस्करण 4** ने `register_menu_items` हटाया और क्रिया पंजीकरण को `register_actions` में एकीकृत
किया.

**संस्करण 2** ने `register_layout_strategies` जोड़ा.

**संस्करण 1** ऐड-ऑन जीवनचक्र, संसाधन पंजीकरण, और UI एकीकरण के लिए कोर हुकों के साथ प्रारंभिक रिलीज़
था.
