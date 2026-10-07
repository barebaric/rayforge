---
description:
  "Rayforge ऐड-ऑन मैनिफेस्ट संदर्भ - अपने ऐड-ऑन का मेटाडेटा, निर्भरताएँ, हुक, और UI योगदान परिभाषित
  करें."
---

# ऐड-ऑन मैनिफेस्ट

प्रत्येक ऐड-ऑन को अपनी जड़ निर्देशिका में एक `rayforge-addon.yaml` फ़ाइल चाहिए. यह मैनिफेस्ट
Rayforge को आपके ऐड-ऑन के बारे में बताता है — उसका नाम, वह क्या देता है, और उसे कैसे लोड करें.

## बुनियादी संरचना

यहाँ सभी सामान्य फ़ील्ड वाला एक पूर्ण मैनिफेस्ट है:

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

## आवश्यक फ़ील्ड

### `name`

आपके ऐड-ऑन के लिए एक अद्वितीय पहचानकर्ता. यह एक वैध Python मॉड्यूल नाम होना चाहिए — केवल अक्षर,
संख्याएँ, और अंडरस्कोर, और यह संख्या से शुरू नहीं हो सकता.

```yaml
name: my_custom_addon
```

### `display_name`

UI में दिखाया गया सुगम नाम. इसमें स्पेस और विशेष वर्ण हो सकते हैं.

```yaml
display_name: "My Custom Addon"
```

### `description`

आपका ऐड-ऑन क्या करता है इसका संक्षिप्त विवरण. यह ऐड-ऑन मैनेजर में दिखाई देता है.

```yaml
description: "Adds support for the XYZ laser cutter."
```

### `api_version`

वह API संस्करण जिसे आपका ऐड-ऑन लक्षित करता है. यह कम से कम 1 (न्यूनतम समर्थित संस्करण) और अधिक से
अधिक वर्तमान संस्करण (9) होना चाहिए. समर्थित से अधिक संस्करण उपयोग करने पर आपका ऐड-ऑन सत्यापन में
विफल हो जाएगा.

```yaml
api_version: 9
```

प्रत्येक संस्करण में क्या बदला इसके लिए [हुक](./addon-hooks.md#api-version-history) दस्तावेज़ीकरण
देखें.

### `author`

ऐड-ऑन लेखक के बारे में जानकारी. `name` फ़ील्ड आवश्यक है; `email` वैकल्पिक है लेकिन उपयोगकर्ताओं के
आपसे संपर्क के लिए अनुशंसित.

```yaml
author:
  name: Jane Doe
  email: jane@example.com
```

## वैकल्पिक फ़ील्ड

### `url`

आपके ऐड-ऑन के होमपेज या रिपॉज़िटरी का URL.

```yaml
url: https://github.com/username/my-custom-addon
```

### `depends`

Rayforge स्वयं के लिए संस्करण बाधाएँ. अपने ऐड-ऑन को चाहिए न्यूनतम संस्करण निर्दिष्ट करें.

```yaml
depends:
  - rayforge>=0.27.0
```

### `requires`

अन्य ऐड-ऑनों पर निर्भरताएँ. संस्करण बाधाओं के साथ ऐड-ऑन नाम सूचीबद्ध करें.

```yaml
requires:
  - some-other-addon>=1.0.0
```

### `version`

आपके ऐड-ऑन का संस्करण नंबर. यह प्रायः git टैग से स्वतः निर्धारित होता है, लेकिन आप इसे स्पष्ट रूप से
निर्दिष्ट कर सकते हैं. सिमेंटिक वर्ज़निंग उपयोग करें (जैसे, `1.0.0`).

```yaml
version: 1.0.0
```

### `maturity`

आपके ऐड-ऑन का परिपक्वता स्तर. उन ऐड-ऑनों के लिए `experimental` उपयोग करें जो अभी पूरे नहीं हैं और
जिनमें अनसुलझी समस्याएँ हो सकती हैं. प्रयोगात्मक ऐड-ऑन ऐड-ऑन मैनेजर में एक समर्पित आइकन के साथ दिखाए
जाते हैं. डिफ़ॉल्ट `stable` है; स्थिर ऐड-ऑनों के लिए फ़ील्ड छोड़ें.

```yaml
maturity: experimental
```

## एंट्री पॉइंट

`provides` खंड परिभाषित करता है कि आपका ऐड-ऑन Rayforge में क्या योगदान देता है.

### Backend

बैकएंड मॉड्यूल मुख्य प्रक्रिया और कार्यकर्ता प्रक्रियाओं दोनों में लोड होता है. इसे मशीन ड्राइवर,
स्टेप प्रकार, ops उत्पादक, और किसी भी कोर कार्यक्षमता के लिए उपयोग करें.

```yaml
provides:
  backend: my_addon.backend
```

मान आपकी ऐड-ऑन निर्देशिका के सापेक्ष एक बिंदु-युक्त Python मॉड्यूल पथ है.

### Frontend

फ़्रंटएंड मॉड्यूल केवल मुख्य प्रक्रिया में लोड होता है. इसे UI घटक, GTK विजेट, और कुछ भी के लिए
उपयोग करें जिसे मुख्य विंडो चाहिए.

```yaml
provides:
  frontend: my_addon.frontend
```

### Assets

आप ऐसी एसेट फ़ाइलें बंडल कर सकते हैं जिन्हें Rayforge पहचानेगा. प्रत्येक एसेट में एक पथ और प्रकार
होता है:

```yaml
provides:
  assets:
    - path: assets/profiles.json
      type: profiles
    - path: assets/templates
      type: templates
```

`path` आपकी ऐड-ऑन जड़ के सापेक्ष है और मौजूद होना चाहिए. एसेट प्रकार Rayforge द्वारा परिभाषित होते
हैं और मशीन प्रोफ़ाइल, सामग्री लाइब्रेरी, या टेम्पलेट जैसी चीज़ें शामिल कर सकते हैं.

## लाइसेंस जानकारी

`license` फ़ील्ड वर्णित करता है कि आपका ऐड-ऑन कैसे लाइसेंसित है. मुफ़्त ऐड-ऑनों के लिए, बस SPDX
पहचानकर्ता का उपयोग करके लाइसेंस नाम निर्दिष्ट करें:

```yaml
license:
  name: MIT
```

सामान्य SPDX पहचानकर्ताओं में `MIT`, `Apache-2.0`, `GPL-3.0`, और `BSD-3-Clause` शामिल हैं.

## भुगतान किए गए ऐड-ऑन

Rayforge Gumroad लाइसेंस सत्यापन के माध्यम से भुगतान किए गए ऐड-ऑनों का समर्थन करता है. यदि आप अपना
ऐड-ऑन बेचना चाहते हैं, तो आप इसे कार्य करने से पहले एक वैध लाइसेंस माँगने के लिए कॉन्फ़िगर कर सकते
हैं.

### बुनियादी भुगतान कॉन्फ़िगरेशन

```yaml
license:
  name: BSL-1.1
  required: true
  purchase_url: https://gum.co/my-addon
```

जब `required` true हो, Rayforge आपका ऐड-ऑन लोड करने से पहले एक वैध लाइसेंस जाँचेगा. `purchase_url`
उन उपयोगकर्ताओं को दिखाया जाता है जिनके पास लाइसेंस नहीं है.

### Gumroad प्रोडक्ट ID

लाइसेंस सत्यापन सक्षम करने के लिए अपनी Gumroad प्रोडक्ट ID जोड़ें:

```yaml
license:
  name: BSL-1.1
  required: true
  purchase_url: https://gum.co/my-addon
  product_id: "abc123def456"
```

कई प्रोडक्ट ID के लिए (जैसे, भिन्न मूल्य स्तर):

```yaml
license:
  name: BSL-1.1
  required: true
  purchase_url: https://gum.co/my-addon
  product_ids:
    - "abc123def456"
    - "xyz789ghi012"
```

### पूर्ण भुगतान ऐड-ऑन उदाहरण

यहाँ एक भुगतान किए गए ऐड-ऑन के लिए एक पूर्ण मैनिफेस्ट है:

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

### कोड में लाइसेंस स्थिति जाँचना

अपने ऐड-ऑन कोड में, आप जाँच सकते हैं कि लाइसेंस वैध है या नहीं:

```python
@hookimpl
def rayforge_init(context):
    if context.license_validator:
        # Check if user has a valid license for your product
        is_valid = context.license_validator.is_product_valid("your_product_id")
        if not is_valid:
            # Optionally show a message or limit functionality
            logger.warning("License not found - some features disabled")
```

## सत्यापन नियम

ऐड-ऑन लोड करते समय Rayforge आपका मैनिफेस्ट सत्यापित करता है. यहाँ नियम हैं:

`name` एक वैध Python पहचानकर्ता होना चाहिए (अक्षर, संख्याएँ, अंडरस्कोर, कोई अग्रग संख्या नहीं).
`api_version` 1 और वर्तमान संस्करण के बीच एक पूर्णांक होना चाहिए. `author.name` रिक्त नहीं हो सकता
या "your-github-username" जैसा प्लेसहोल्डर टेक्स्ट नहीं रख सकता. एंट्री पॉइंट वैध मॉड्यूल पथ होने
चाहिए और मॉड्यूल मौजूद होने चाहिए. एसेट पथ सापेक्ष होने चाहिए (कोई `..` या अग्रग `/` नहीं) और
फ़ाइलें मौजूद होनी चाहिए.

यदि सत्यापन विफल होता है, तो Rayforge एक त्रुटि लॉग करता है और आपका ऐड-ऑन छोड़ देता है. इन समस्याओं
को पकड़ने के लिए विकास के दौरान कंसोल आउटपुट जाँचें.
