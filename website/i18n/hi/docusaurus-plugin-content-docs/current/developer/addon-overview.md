---
description:
  "Rayforge के लिए ऐड-ऑन विकसित करें. ऐड-ऑन प्रणाली, हुक, मैनिफेस्ट, और कस्टम कार्यक्षमता से
  Rayforge विस्तारित करने का तरीका सीखें."
---

# ऐड-ऑन विकास अवलोकन

Rayforge [pluggy](https://pluggy.readthedocs.io/) पर आधारित एक ऐड-ऑन प्रणाली उपयोग करता है जो आपको
कोर कोडबेस संशोधित किए बिना कार्यक्षमता विस्तारित करने, नए मशीन ड्राइवर जोड़ने, या कस्टम तर्क एकीकृत
करने देता है.

## त्वरित प्रारंभ

शुरुआत करने का सबसे तेज़ तरीका आधिकारिक
[rayforge-addon-template](https://github.com/barebaric/rayforge-addon-template) है. उसे फ़ोर्क या
क्लोन करें, निर्देशिका का नाम बदलें, और मेटाडेटा को अपने ऐड-ऑन से मेल खाने के लिए अपडेट करें.

## ऐड-ऑन कैसे काम करते हैं

`AddonManager` वैध ऐड-ऑनों के लिए `addons` निर्देशिका स्कैन करता है. ऐड-ऑन केवल आपके Python कोड के
साथ एक `rayforge-addon.yaml` मैनिफेस्ट फ़ाइल युक्त एक निर्देशिका है.

एक सामान्य ऐड-ऑन ऐसा दिखता है:

```text
my-rayforge-addon/
├── rayforge-addon.yaml  <-- Required manifest
├── my_addon/            <-- Your Python package
│   ├── __init__.py
│   ├── backend.py       <-- Backend entry point
│   └── frontend.py      <-- Frontend entry point (optional)
├── assets/              <-- Optional resources
├── locales/             <-- Optional translations (.po files)
└── README.md
```

## आपका पहला ऐड-ऑन

चलिए एक सरल ऐड-ऑन बनाते हैं जो एक कस्टम मशीन ड्राइवर पंजीकृत करता है. पहले, मैनिफेस्ट बनाएँ:

```yaml title="rayforge-addon.yaml"
name: my_laser_driver
display_name: "My Laser Driver"
description: "Adds support for the XYZ laser cutter."
api_version: 9

author:
  name: Jane Doe
  email: jane@example.com

provides:
  backend: my_addon.backend
```

अब अपना ड्राइवर पंजीकृत करने वाला बैकएंड मॉड्यूल बनाएँ:

```python title="my_addon/backend.py"
import pluggy

hookimpl = pluggy.HookimplMarker("rayforge")

@hookimpl
def register_machines(machine_manager):
    """Register our custom machine driver."""
    from .my_driver import MyLaserMachine
    machine_manager.register("my_laser", MyLaserMachine)
```

बस इतना ही! Rayforge शुरू होने पर आपका ऐड-ऑन लोड हो जाएगा, और आपका मशीन ड्राइवर उपयोगकर्ताओं को
उपलब्ध हो जाएगा.

[मैनिफेस्ट](./addon-manifest.md) दस्तावेज़ीकरण सभी उपलब्ध कॉन्फ़िगरेशन विकल्प कवर करता है.

## एंट्री पॉइंट समझना

ऐड-ऑन दो एंट्री पॉइंट दे सकते हैं, प्रत्येक भिन्न समय पर लोड होता है:

**बैकएंड** एंट्री पॉइंट मुख्य प्रक्रिया और कार्यकर्ता प्रक्रियाओं दोनों में लोड होता है. इसे मशीन
ड्राइवर, स्टेप प्रकार, ops उत्पादक और ट्रांसफ़ॉर्मर, या किसी भी कोर कार्यक्षमता के लिए उपयोग करें
जिसे UI निर्भरताओं की आवश्यकता नहीं होती.

**फ़्रंटएंड** एंट्री पॉइंट केवल मुख्य प्रक्रिया में लोड होता है. यहाँ आप UI घटक, GTK विजेट, मेनू
आइटम, और कुछ भी रखेंगे जिसे मुख्य विंडो तक पहुँच चाहिए.

दोनों `my_addon.backend` जैसे बिंदु-युक्त मॉड्यूल पथों के रूप में निर्दिष्ट होते हैं.

## हुक के साथ Rayforge से जुड़ना

Rayforge ऐड-ऑनों को एप्लिकेशन के साथ एकीकृत करने देने के लिए `pluggy` हुक उपयोग करता है. बस अपने
फ़ंक्शनों को `@pluggy.HookimplMarker("rayforge")` से सजाएँ:

```python
import pluggy
from rayforge.context import RayforgeContext

hookimpl = pluggy.HookimplMarker("rayforge")

@hookimpl
def rayforge_init(context: RayforgeContext):
    """Called when Rayforge is fully initialized."""
    # Your setup code here
    pass

@hookimpl
def on_unload():
    """Called when the addon is being disabled or unloaded."""
    # Clean up resources here
    pass
```

[हुक](./addon-hooks.md) दस्तावेज़ीकरण हर उपलब्ध हुक और कब वह कॉल होता है वर्णित करता है.

## अपने घटक पंजीकृत करना

अधिकांश हुक एक रजिस्ट्री ऑब्जेक्ट प्राप्त करते हैं जिसका उपयोग आप अपने कस्टम घटक पंजीकृत करने के लिए
करते हैं:

```python
@hookimpl
def register_steps(step_registry):
    from .my_step import MyCustomStep
    step_registry.register(MyCustomStep)

@hookimpl
def register_actions(action_registry):
    from .actions import setup_actions
    setup_actions(action_registry)
```

[रजिस्ट्री](./addon-registries.md) दस्तावेज़ीकरण प्रत्येक रजिस्ट्री और उनका उपयोग कैसे करें समझाता
है.

## Rayforge के डेटा तक पहुँच {#accessing-rayforges-data}

`rayforge_init` हुक आपको एक `RayforgeContext` ऑब्जेक्ट तक पहुँच देता है. इस संदर्भ के माध्यम से, आप
Rayforge की हर चीज़ तक पहुँच सकते हैं:

आप `context.machine` द्वारा वर्तमान में सक्रिय मशीन पा सकते हैं, या सभी मशीनों तक
`context.machine_mgr` द्वारा पहुँच सकते हैं. `context.config` ऑब्जेक्ट वैश्विक सेटिंग्स रखता है,
जबकि `context.camera_mgr` कैमरा फ़ीड तक पहुँच देता है. सामग्रियों के लिए `context.material_mgr`
उपयोग करें, और प्रोसेसिंग रेसिपियों के लिए `context.recipe_mgr` उपयोग करें. G-code डायलेक्ट मैनेजर
`context.dialect_mgr` के रूप में उपलब्ध है, और AI सुविधाएँ `context.ai_provider_mgr` के माध्यम से
जाती हैं. स्थानीयकरण के लिए, वर्तमान भाषा कोड के लिए `context.language` जाँचें. ऐड-ऑन मैनेजर स्वयं
`context.addon_mgr` के रूप में उपलब्ध है, और यदि आप भुगतान किए गए ऐड-ऑन बना रहे हैं, तो
`context.license_validator` लाइसेंस सत्यापन संभालता है.

## अनुवाद जोड़ना

ऐड-ऑन मानक `.po` फ़ाइलों का उपयोग करके अनुवाद दे सकते हैं. उन्हें इस तरह व्यवस्थित करें:

```text
my-rayforge-addon/
├── locales/
│   ├── de/
│   │   └── LC_MESSAGES/
│   │       └── my_addon.po
│   └── es/
│       └── LC_MESSAGES/
│           └── my_addon.po
```

आपका ऐड-ऑन लोड होने पर Rayforge `.po` फ़ाइलों को स्वतः `.mo` फ़ाइलों में कंपाइल करता है.

## विकास के दौरान परीक्षण

अपने ऐड-ऑन को स्थानीय रूप से परीक्षण करने के लिए, अपने विकास फ़ोल्डर से Rayforge के addons
निर्देशिका तक एक प्रतीकात्मक लिंक बनाएँ.

पहले, अपनी कॉन्फ़िगरेशन निर्देशिका ढूँढ़ें. Windows पर, वह
`C:\Users\<User>\AppData\Local\rayforge\rayforge\addons` है. macOS पर,
`~/Library/Application Support/rayforge/addons` में देखें. Linux पर, वह `~/.config/rayforge/addons`
है.

फिर सिम्लिंक बनाएँ:

```bash
ln -s /path/to/my-rayforge-addon ~/.config/rayforge/addons/my-rayforge-addon
```

Rayforge पुनरारंभ करें और `Loaded addon: my_laser_driver` जैसे संदेश के लिए कंसोल जाँचें.

## अपना ऐड-ऑन साझा करें

अपना ऐड-ऑन साझा करने के लिए तैयार होने पर, उसे GitHub या GitLab पर एक सार्वजनिक Git रिपॉज़िटरी में
पुश करें. फिर रिपॉज़िटरी फ़ोर्क करके, अपने ऐड-ऑन का मेटाडेटा जोड़कर, और पुल अनुरोध खोलकर उसे
[rayforge-registry](https://github.com/barebaric/rayforge-registry) में सबमिट करें.

स्वीकृत होने के बाद, उपयोगकर्ता आपका ऐड-ऑन सीधे Rayforge के ऐड-ऑन मैनेजर के माध्यम से इंस्टॉल कर
सकते हैं.
