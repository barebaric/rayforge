---
description: "Rayforge के लिए कमांड-लाइन इंटरफ़ेस संदर्भ."
---

# कमांड लाइन

Rayforge के कमांड-लाइन विकल्पों का पूर्ण संदर्भ.

```
rayforge [options] [filenames...]
```

---

## स्थितिगत तर्क

| तर्क        | विवरण                                     |
| ----------- | ----------------------------------------- |
| `filenames` | लॉन्च पर खोलने के लिए SVG या छवि फ़ाइलें. |

---

## विकल्प

| विकल्प              | विवरण                               |
| ------------------- | ----------------------------------- |
| `--version`         | संस्करण प्रिंट करके बाहर निकलें.    |
| `-h`, `--help`      | सहायता दिखाकर बाहर निकलें.          |
| `--loglevel LEVEL`  | लॉगिंग स्तर. डिफ़ॉल्ट: `INFO`.      |
| `--config DIR`      | कस्टम कॉन्फ़िग निर्देशिका.          |
| `--exit`            | आयात स्थिर होने के बाद बाहर निकलें. |
| `--vector`          | सीधा वेक्टर आयात बल दें.            |
| `--trace`           | बिटमैप ट्रेस आयात बल दें.           |
| `--script SCRIPT`   | जल्दी स्टार्टअप स्क्रिप्ट.          |
| `--uiscript SCRIPT` | UI स्क्रिप्ट (लोड के बाद).          |

---

## उदाहरण

### फ़ाइल खोलें

```bash
rayforge myproject.ryp
```

### कई फ़ाइलें खोलें

```bash
rayforge part1.svg logo.png design.ryp
```

### ट्रेसिंग के साथ आयात करें

```bash
rayforge --trace photo.png
```

### जल्दी स्क्रिप्ट चलाकर बाहर निकलें

```bash
rayforge --exit --script register_functions.py \
    myproject.ryp
```

### UI स्क्रिप्ट चलाएँ (स्वचालन)

```bash
rayforge --exit --uiscript screenshot.py \
    myproject.ryp
```

### बैच निर्यात

```bash
rayforge --exit --vector input.svg
```

---

## जल्दी स्क्रिप्ट (`--script`)

`--script` ध्वज एक Python स्क्रिप्ट **स्टार्टअप के दौरान समकालिक रूप से** चलाता है, ऐड-ऑन लोड होने
से पहले और मुख्य विंडो बनने से पहले. इससे वह इनके लिए सही जगह बन जाता है:

- `pluggy` प्लगइन मैनेजर के साथ प्लगइन पंजीकृत करना
- एप्लिकेशन संदर्भ कॉन्फ़िगर करना
- टेक्स्ट बॉक्सों के लिए टेम्पलेट फ़ंक्शन पंजीकृत करना
- ऐप शुरू होने से पहले पर्यावरण चर सेट करना

स्क्रिप्ट `get_context()` के माध्यम से संदर्भ तक पहुँचती है:

```python
from rayforge.context import get_context

ctx = get_context()
# Register plugins, configure services, etc.
```

### उदाहरण: कस्टम टेम्पलेट फ़ंक्शन पंजीकृत करें

```python
"""Register a custom function for text box expressions.

Run with: rayforge --script register_fn.py
"""
from rayforge.context import get_context
from sketcher.core.template_functions import (
    register_template_function,
)

register_template_function("myid", lambda: "PART-001")
```

अब `{myid()}` किसी भी टेक्स्ट बॉक्स में काम करता है.

पूर्ण ट्यूटोरियल के लिए स्केचर दस्तावेज़ में
[कस्टम टेम्पलेट फ़ंक्शन](../features/sketcher/expressions.md#custom-template-functions) देखें.

---

## UI स्क्रिप्ट (`--uiscript`)

`--uiscript` ध्वज एक Python स्क्रिप्ट **मुख्य विंडो पूरी तरह मैप और लोड होने के बाद**, पृष्ठभूमि
थ्रेड में चलाता है. इससे वह इनके लिए सही जगह बन जाता है:

- स्वचालित UI परीक्षण
- एप्लिकेशन के स्क्रीनशॉट लेना
- एंड-टू-एंड वर्कफ़्लो चलाना

स्क्रिप्ट एप्लिकेशन और विंडो को सीधे आयात कर सकती है:

```python
from rayforge.uiscript import app, win
```

स्क्रिप्ट **पृष्ठभूमि थ्रेड** में चलती है — GTK विजेट एक्सेस करते समय थ्रेड सुरक्षा का ध्यान रखें
(GTK ऑपरेशनों के लिए `GLib.idle_add` उपयोग करें).

### उदाहरण: स्क्रीनशॉट लें

```python
"""Capture a screenshot of the main window."""
from rayforge.uiscript import app, win

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import GLib

def capture():
    surface = win.get_surface()
    if surface:
        surface.write_to_png("/tmp/rayforge_screenshot.png")
    return GLib.SOURCE_REMOVE

GLib.idle_add(capture)
```

---

## दोनों ध्वज उपयोग करना

`--script` और `--uiscript` दोनों साथ उपयोग किए जा सकते हैं. `--script` पहले (समकालिक रूप से) चलता
है, फिर विंडो लोड होती है, और फिर `--uiscript` चलता है:

```bash
rayforge --script early_setup.py \
    --uiscript automation.py \
    myproject.ryp
```

यह तब उपयोगी है जब आपको पहले प्लगइन पंजीकृत करने हों और फिर बाद में UI संचालित करना हो.
