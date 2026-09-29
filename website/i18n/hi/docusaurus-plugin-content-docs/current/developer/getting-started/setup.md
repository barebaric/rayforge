---
description: "Rayforge विकास वातावरण सेट करें. स्रोत कोड से एप्लिकेशन बनाएँ, चलाएँ, और डिबग करें."
---

# सेटअप

यह गाइड Rayforge के लिए अपना विकास वातावरण सेट करना कवर करती है.

## Linux

### पूर्वापेक्षाएँ

Pixi इंस्टॉलेशन निर्देशों के लिए
[इंस्टॉलेशन गाइड](../../getting-started/installation.mdx#linux-pixi) देखें.

### Pre-commit हुक (वैकल्पिक)

प्रत्येक कमिट से पहले अपना कोड स्वतः फ़ॉर्मेट और लिंट करने के लिए, आप pre-commit हुक इंस्टॉल कर सकते
हैं:

```bash
pixi run pre-commit-install
```

### उपयोगी कमांड

सभी कमांड `pixi run` द्वारा चलती हैं:

- `pixi run rayforge`: एप्लिकेशन चलाएँ.
  - अधिक वर्बोज़ आउटपुट के लिए `--loglevel=DEBUG` जोड़ें.
- `pixi run test`: `pytest` के साथ पूर्ण परीक्षण सूट चलाएँ.
- `pixi run format`: `ruff` का उपयोग करके सारा कोड फ़ॉर्मेट करें.
- `pixi run lint`: सभी लिंटर चलाएँ (`flake8`, `pyflakes`, `pyright`).

## macOS

### पूर्वापेक्षाएँ

Pixi इंस्टॉलेशन निर्देशों के लिए
[इंस्टॉलेशन गाइड](../../getting-started/installation.mdx#linux-pixi) देखें. Pixi Apple Silicon
(`osx-arm64`) पर macOS का समर्थन करता है.

### उपयोगी कमांड

सभी कमांड `pixi run` द्वारा चलती हैं, Linux की तरह ही:

- `pixi run rayforge`: एप्लिकेशन चलाएँ.
  - अधिक वर्बोज़ आउटपुट के लिए `--loglevel=DEBUG` जोड़ें.
- `pixi run test`: `pytest` के साथ पूर्ण परीक्षण सूट चलाएँ.
- `pixi run format`: `ruff` का उपयोग करके सारा कोड फ़ॉर्मेट करें.
- `pixi run lint`: सभी लिंटर चलाएँ (`flake8`, `pyflakes`, `pyright`).

## Windows

### पूर्वापेक्षाएँ

विस्तृत MSYS2 डेवलपर सेटअप निर्देशों के लिए
[इंस्टॉलेशन गाइड](../../getting-started/installation.mdx#windows-developer) देखें.

### त्वरित प्रारंभ

Windows पर विकास कार्य `run.bat` स्क्रिप्ट द्वारा प्रबंधित होते हैं, जो MSYS2 शेल के लिए एक रैपर है.

रिपॉज़िटरी क्लोन करने और MSYS2 सेटअप पूरा करने के बाद, आप इन कमांड को मानक Windows Command Prompt या
PowerShell से चला सकते हैं:

```batch
.\run.bat setup
```

यह सभी आवश्यक सिस्टम और Python पैकेज आपके MSYS2/UCRT64 वातावरण में इंस्टॉल करने के लिए
`scripts/win/win_setup.sh` निष्पादित करता है.

### Pre-commit हुक (वैकल्पिक)

प्रत्येक कमिट से पहले अपना कोड स्वतः फ़ॉर्मेट और लिंट करने के लिए, इसे MSYS2 UCRT64 शेल से चलाएँ:

```bash
bash scripts/win/win_setup_dev.sh
```

:::note

Pre-commit हुक के लिए git कमांड MSYS2 UCRT64 शेल के भीतर से चलानी होती हैं, PowerShell या Command
Prompt से नहीं.

:::

### उपयोगी कमांड

सभी कमांड `run.bat` स्क्रिप्ट द्वारा चलती हैं:

- `run app`: स्रोत से एप्लिकेशन चलाएँ.
  - अधिक वर्बोज़ आउटपुट के लिए `--loglevel=DEBUG` जोड़ें.
- `run test`: `pytest` का उपयोग करके पूर्ण परीक्षण सूट चलाएँ.
- `run lint`: सभी लिंटर चलाएँ (`flake8`, `pyflakes`, `pyright`).
- `run format`: `ruff` का उपयोग करके कोड फ़ॉर्मेट और स्वतः-ठीक करें.
- `run build`: अंतिम Windows निष्पादन योग्य (`.exe`) बनाएँ.

वैकल्पिक रूप से, आप स्क्रिप्ट सीधे MSYS2 UCRT64 शेल से चला सकते हैं:

- `bash scripts/win/win_run.sh`: एप्लिकेशन चलाएँ.
- `bash scripts/win/win_test.sh`: परीक्षण सूट चलाएँ.
- `bash scripts/win/win_lint.sh`: सभी लिंटर चलाएँ.
- `bash scripts/win/win_format.sh`: कोड फ़ॉर्मेट और स्वतः-ठीक करें.
- `bash scripts/win/win_build.sh`: Windows निष्पादन योग्य बनाएँ.
