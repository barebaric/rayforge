---
description:
  "Rayforge में Snap पैकेज अनुमति समस्याएँ ठीक करें. Linux सिस्टमों पर USB सीरियल पोर्ट और कैमरा
  पहुँच प्रदान करें."
---

# Snap अनुमतियाँ (Linux)

यह पृष्ठ समझाता है कि Linux पर Snap पैकेज के रूप में इंस्टॉल होने पर Rayforge के लिए अनुमतियाँ कैसे
कॉन्फ़िगर करें.

## Snap अनुमतियाँ क्या हैं?

Snap संदरित एप्लिकेशन हैं जो सुरक्षा के लिए एक सैंडबॉक्स में चलते हैं. डिफ़ॉल्ट रूप से, उनकी सिस्टम
संसाधनों तक सीमित पहुँच होती है. कुछ सुविधाओं (लेज़र नियंत्रकों के सीरियल पोर्ट जैसी) का उपयोग करने
के लिए, आपको स्पष्ट रूप से अनुमतियाँ प्रदान करनी होंगी.

## आवश्यक अनुमतियाँ

पूर्ण कार्यक्षमता के लिए Rayforge को ये Snap इंटरफ़ेस जुड़े चाहिए:

| इंटरफ़ेस          | उद्देश्य                                        | आवश्यक?                        |
| ----------------- | ----------------------------------------------- | ------------------------------ |
| `serial-port`     | USB सीरियल डिवाइसों (लेज़र नियंत्रकों) तक पहुँच | **हाँ** (मशीन नियंत्रण के लिए) |
| `home`            | आपकी होम निर्देशिका में फ़ाइलें पढ़ें/लिखें     | स्वतः जुड़ा                    |
| `removable-media` | बाहरी ड्राइव और USB भंडारण तक पहुँच             | वैकल्पिक                       |
| `network`         | नेटवर्क कनेक्टिविटी (अपडेट आदि के लिए)          | स्वतः जुड़ा                    |

---

## सीरियल पोर्ट पहुँच प्रदान करना

**Rayforge के लिए यह सबसे महत्वपूर्ण अनुमति है.**

### पूर्वापेक्षा: dialout समूह सदस्यता

Debian-आधारित वितरणों पर, Snap पैकेज का उपयोग करते समय भी आपके उपयोगकर्ता को `dialout` समूह का सदस्य
होना चाहिए. इस समूह सदस्यता के बिना, सीरियल पोर्ट एक्सेस करने की कोशिश करते समय आपको AppArmor DENIED
संदेश मिल सकते हैं.

```bash
# Add your user to the dialout group
sudo usermod -a -G dialout $USER
```

**महत्वपूर्ण:** समूह बदलाव लागू होने के लिए आपको लॉग आउट करके वापस लॉग इन करना होगा (या रीबूट करना
होगा).

### वर्तमान अनुमतियाँ जाँचें

```bash
# View all connections for Rayforge
snap connections rayforge
```

`serial-port` इंटरफ़ेस देखें. यदि वह "disconnected" या "-" दिखाता है, तो आपको उसे जोड़ना होगा.

### सीरियल पोर्ट इंटरफ़ेस जोड़ें

```bash
# Grant serial port access
sudo snap connect rayforge:serial-port
```

**आपको यह केवल एक बार करना है.** अनुमति ऐप अपडेट और रीबूट में बनी रहती है.

### कनेक्शन सत्यापित करें

```bash
# Check if serial-port is now connected
snap connections rayforge | grep serial-port
```

अपेक्षित आउटपुट:

```
serial-port     rayforge:serial-port     :serial-port     -
```

यदि आपको प्लग/स्लॉट सूचक दिखे, तो कनेक्शन सक्रिय है.

---

## रिमूवेबल मीडिया पहुँच प्रदान करना

यदि आप USB ड्राइवों या बाहरी भंडारण से फ़ाइलें आयात/निर्यात करना चाहते हैं:

```bash
# Grant access to removable media
sudo snap connect rayforge:removable-media
```

अब आप `/media` और `/mnt` में फ़ाइलें एक्सेस कर सकते हैं.

---

## Snap अनुमति समस्या निवारण

### सीरियल पोर्ट अभी भी काम नहीं कर रहा

**इंटरफ़ेस जोड़ने के बाद:**

1. **USB डिवाइस दोबारा लगाएँ:**
   - अपना लेज़र नियंत्रक अनप्लग करें
   - 5 सेकंड प्रतीक्षा करें
   - इसे वापस लगाएँ

2. **Rayforge पुनरारंभ करें:**
   - Rayforge पूरी तरह बंद करें
   - एप्लिकेशन मेनू से फिर से लॉन्च करें या:
     ```bash
     snap run rayforge
     ```

3. **जाँचें कि पोर्ट दिखाई देता है:**
   - Rayforge Settings Machine खोलें
   - ड्रॉपडाउन में सीरियल पोर्ट देखें
   - `/dev/ttyUSB0`, `/dev/ttyACM0`, या समान दिखना चाहिए

4. **सत्यापित करें कि डिवाइस मौजूद है:**
   ```bash
   # List USB serial devices
   ls -l /dev/ttyUSB* /dev/ttyACM*
   ```

### जुड़े इंटरफ़ेस के बावजूद "अनुमति अस्वीकृत"

यह दुर्लभ है लेकिन तब हो सकता है जब:

1. **Snap इंस्टॉलेशन टूटा हुआ है:**

   ```bash
   # Reinstall the snap
   sudo snap refresh rayforge --devmode
   # Or if that fails:
   sudo snap remove rayforge
   sudo snap install rayforge
   # Re-connect interfaces
   sudo snap connect rayforge:serial-port
   ```

2. **टकराते udev नियम:**
   - कस्टम सीरियल पोर्ट नियमों के लिए `/etc/udev/rules.d/` जाँचें
   - वे Snap की डिवाइस पहुँच के साथ टकरा सकते हैं

3. **AppArmor अस्वीकृतियाँ:**

   ```bash
   # Check for AppArmor denials
   sudo journalctl -xe | grep DENIED | grep rayforge
   ```

   यदि आपको सीरियल पोर्टों के लिए अस्वीकृतियाँ दिखें, तो AppArmor प्रोफ़ाइल टकराव हो सकता है.

### होम निर्देशिका के बाहर फ़ाइलें एक्सेस नहीं हो सकतीं

**डिज़ाइन द्वारा**, जब तक आप `removable-media` प्रदान न करें, Snap आपकी होम निर्देशिका के बाहर की
फ़ाइलें एक्सेस नहीं कर सकते.

**समाधान विकल्प:**

1. **फ़ाइलें अपनी होम निर्देशिका में ले जाएँ:**

   ```bash
   # Copy SVG files to ~/Documents
   cp /some/other/location/*.svg ~/Documents/
   ```

2. **रिमूवेबल मीडिया पहुँच प्रदान करें:**

   ```bash
   sudo snap connect rayforge:removable-media
   ```

3. **Snap का फ़ाइल चयनकर्ता उपयोग करें:**
   - अंतर्निहित फ़ाइल चयनकर्ता की व्यापक पहुँच होती है
   - फ़ाइलें कमांड-लाइन तर्कों के बजाय File Open द्वारा खोलें

---

## मैन्युअल इंटरफ़ेस प्रबंधन

### सभी उपलब्ध इंटरफ़ेस सूचीबद्ध करें

```bash
# See all Snap interfaces on your system
snap interface
```

### इंटरफ़ेस डिस्कनेक्ट करें

```bash
# Disconnect serial-port (if needed)
sudo snap disconnect rayforge:serial-port
```

### डिस्कनेक्ट के बाद पुनः जोड़ें

```bash
sudo snap connect rayforge:serial-port
```

---

## विकल्प: स्रोत से इंस्टॉल करें

यदि Snap अनुमतियाँ आपके वर्कफ़्लो के लिए बहुत प्रतिबंधक हैं:

**विकल्प 1: स्रोत से बनाएँ**

```bash
# Clone the repository
git clone https://github.com/kylemartin57/rayforge.git
cd rayforge

# Install dependencies using pixi
pixi install

# Run Rayforge
pixi run rayforge
```

**लाभ:**

- कोई अनुमति प्रतिबंध नहीं
- पूर्ण सिस्टम पहुँच
- डिबगिंग आसान
- नवीनतम विकास संस्करण

**हानि:**

- मैन्युअल अपडेट (git pull)
- प्रबंधन के लिए अधिक निर्भरताएँ
- कोई स्वतः अपडेट नहीं

**विकल्प 2: Flatpak उपयोग करें (यदि उपलब्ध हो)**

Flatpak में समान सैंडबॉक्सिंग है लेकिन कभी-कभी भिन्न अनुमति मॉडल के साथ. जाँचें कि Rayforge Flatpak
पैकेज देता है या नहीं.

---

## Snap अनुमति सर्वोत्तम प्रथाएँ

### केवल आवश्यक जोड़ें

जिन इंटरफ़ेसों का उपयोग नहीं करते उन्हें न जोड़ें:

-  यदि लेज़र नियंत्रक उपयोग करते हैं तो `serial-port` जोड़ें
-  यदि USB ड्राइवों से आयात करते हैं तो `removable-media` जोड़ें
- L "संभावना के लिए" सब कुछ न जोड़ें - सुरक्षा उद्देश्य को पराजित करता है

### Snap स्रोत सत्यापित करें

हमेशा आधिकारिक Snap Store से इंस्टॉल करें:

```bash
# Check publisher
snap info rayforge
```

यह देखें:

- सत्यापित प्रकाशक
- आधिकारिक रिपॉज़िटरी स्रोत
- नियमित अपडेट

---

## Snap सैंडबॉक्स समझना

### Snap डिफ़ॉल्ट रूप से क्या एक्सेस कर सकते हैं?

**अनुमत:**

- आपकी होम निर्देशिका की फ़ाइलें
- नेटवर्क कनेक्शन
- डिस्प्ले/ऑडियो

**स्पष्ट अनुमति के बिना अनुमत नहीं:**

- सीरियल पोर्ट (USB डिवाइस)
- रिमूवेबल मीडिया
- सिस्टम फ़ाइलें
- अन्य उपयोगकर्ताओं की होम निर्देशिकाएँ

### यह Rayforge के लिए क्यों मायने रखता है

Rayforge को चाहिए:

1. **होम निर्देशिका पहुँच** (स्वतः प्रदान)
   - परियोजना फ़ाइलें सहेजने के लिए
   - आयातित SVG/DXF फ़ाइलें पढ़ने के लिए
   - वरीयताएँ संग्रहीत करने के लिए

2. **सीरियल पोर्ट पहुँच** (प्रदान की जानी चाहिए)
   - लेज़र नियंत्रकों से संवाद करने के लिए
   - **यह महत्वपूर्ण अनुमति है**

3. **रिमूवेबल मीडिया** (वैकल्पिक)
   - USB ड्राइवों से फ़ाइलें आयात करने के लिए
   - G-code बाहरी भंडारण में निर्यात करने के लिए

---

## Snap समस्याएँ डिबग करें

### वर्बोज़ Snap लॉगिंग सक्षम करें

```bash
# Run Snap with debug output
snap run --shell rayforge
# Inside the snap shell:
export RAYFORGE_LOG_LEVEL=DEBUG
exec rayforge
```

### Snap लॉग जाँचें

```bash
# View Rayforge logs
snap logs rayforge

# Follow logs in real-time
snap logs -f rayforge
```

### अस्वीकृतियों के लिए सिस्टम जर्नल जाँचें

```bash
# Look for AppArmor denials
sudo journalctl -xe | grep DENIED | grep rayforge

# Look for USB device events
sudo journalctl -f -u snapd
# Then plug in your laser controller
```

---

## सहायता प्राप्त करें

यदि आपको अभी भी Snap-संबंधी समस्याएँ हैं:

1. **पहले अनुमतियाँ जाँचें:**

   ```bash
   snap connections rayforge
   ```

2. **सीरियल पोर्ट परीक्षण का प्रयास करें:**

   ```bash
   # If you have screen or minicom installed
   sudo snap connect rayforge:serial-port
   # Then test in Rayforge
   ```

3. **समस्या इनके साथ रिपोर्ट करें:**
   - `snap connections rayforge` का आउटपुट
   - `snap version` का आउटपुट
   - `snap info rayforge` का आउटपुट
   - आपका Ubuntu/Linux वितरण संस्करण
   - सटीक त्रुटि संदेश

4. **विकल्पों पर विचार करें:**
   - स्रोत से इंस्टॉल करें (ऊपर देखें)
   - भिन्न पैकेज फ़ॉर्मेट उपयोग करें (AppImage, Flatpak)

---

## त्वरित संदर्भ कमांड

```bash
# Grant serial port access (most important)
sudo snap connect rayforge:serial-port

# Grant removable media access
sudo snap connect rayforge:removable-media

# Check current connections
snap connections rayforge

# View Rayforge logs
snap logs rayforge

# Refresh/update Rayforge
sudo snap refresh rayforge

# Remove and reinstall (last resort)
sudo snap remove rayforge
sudo snap install rayforge
sudo snap connect rayforge:serial-port
```

---

## संबंधित पृष्ठ

- [कनेक्शन समस्याएँ](connection) - सीरियल कनेक्शन समस्या निवारण
- [डिबग मोड](debug) - नैदानिक लॉगिंग सक्षम करें
- [इंस्टॉलेशन](../getting-started/installation.mdx) - इंस्टॉलेशन गाइड
- [सामान्य सेटिंग्स](../machine/general.md) - मशीन सेटअप
