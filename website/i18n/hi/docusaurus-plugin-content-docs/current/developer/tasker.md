---
description:
  "Rayforge टास्कर प्रणाली - पाथ अनुकूलन और G-code जनरेशन जैसे लंबे समय चलने वाले ऑपरेशनों के लिए
  पृष्ठभूमि कार्य प्रबंधन."
---

# टास्कर: पृष्ठभूमि कार्य प्रबंधन

`tasker` GTK एप्लिकेशन की पृष्ठभूमि में लंबे समय चलने वाले कार्य बिना UI जमाए चलाने के लिए एक
मॉड्यूल है. यह I/O-बाउंड (`asyncio`) और CPU-बाउंड कार्य दोनों के लिए एक सरल, एकीकृत API देता है.

> **मल्टीप्रोसेसिंग पर नोट.** सबप्रोसेस पूल डिफ़ॉल्ट रूप से सुप्त रहता है — वह पहली `run_process()`
> कॉल पर आलसी रूप से शुरू होता है. उत्पादन पाइपलाइन कार्य इसके बजाय
> [raygeo](https://github.com/barebaric/raygeo) का भीतरी rayon थ्रेड पूल उपयोग करते हैं. नीचे के
> `run_process` उदाहरण भविष्य के उपयोग के लिए अग्र-दृष्टि संदर्भ के रूप में वैध रहते हैं.

## कोर अवधारणाएँ

1. **`task_mgr`**: वैश्विक सिंगलटन प्रॉक्सी जिसका उपयोग आप सभी कार्य प्रारंभ और रद्द करने के लिए
   करते हैं
2. **`Task`**: एक एकल पृष्ठभूमि जॉब दर्शाने वाला ऑब्जेक्ट. स्थिति ट्रैक करने के लिए इसका उपयोग करें
3. **`ExecutionContext` (`context`**): एक ऑब्जेक्ट जो आपके पृष्ठभूमि फ़ंक्शन को पहले तर्क के रूप में
   पारित होता है. आपका कोड प्रगति रिपोर्ट करने, संदेश भेजने, और रद्दीकरण जाँचने के लिए इसका उपयोग
   करता है
4. **`TaskManagerProxy`**: एक थ्रेड-सुरक्षित प्रॉक्सी जो कॉल मुख्य थ्रेड में चल रहे वास्तविक
   TaskManager को अग्रेषित करता है

## त्वरित प्रारंभ

सभी पृष्ठभूमि कार्य वैश्विक `task_mgr` द्वारा प्रबंधित होते हैं.

### I/O-बाउंड कार्य चलाना (जैसे, नेटवर्क, फ़ाइल पहुँच)

`async` फ़ंक्शनों के लिए `add_coroutine` उपयोग करें. ये हल्के होते हैं और I/O की प्रतीक्षा करने वाले
कार्यों के लिए आदर्श हैं.

```python
import asyncio
from rayforge.shared.tasker import task_mgr

# Your background function MUST accept `context` as the first argument.
async def my_io_task(context, url):
    context.set_message("Downloading...")
    # ... perform async download ...
    await asyncio.sleep(2) # Simulate work
    context.set_progress(1.0)
    context.set_message("Download complete!")

# Start the task from your UI code (e.g., a button click)
task_mgr.add_coroutine(my_io_task, "http://example.com", key="downloader")
```

### CPU-बाउंड कार्य चलाना (जैसे, भारी गणना)

सामान्य फ़ंक्शनों के लिए `run_process` उपयोग करें. ये GIL से बचने और UI उत्तरदायी रखने के लिए एक अलग
प्रक्रिया में चलते हैं.

```python
import time
from rayforge.shared.tasker import task_mgr

# A regular function, not async.
def my_cpu_task(context, iterations):
    context.set_total(iterations)
    context.set_message("Calculating...")
    for i in range(iterations):
        # ... perform heavy calculation ...
        time.sleep(0.1) # Simulate work
        context.set_progress(i + 1)
    return "Final Result"

# Start the task
task_mgr.run_process(my_cpu_task, 50, key="calculator")
```

### थ्रेड-बाउंड कार्य चलाना

उन कार्यों के लिए `run_thread` उपयोग करें जो थ्रेड में चलने चाहिए लेकिन पूर्ण प्रक्रिया पृथक्करण की
आवश्यकता नहीं रखते. यह मेमोरी साझा करने वाले लेकिन फिर भी UI अवरोधित न करने वाले कार्यों के लिए
उपयोगी है.

```python
import time
from rayforge.shared.tasker import task_mgr

# A regular function that will run in a thread
def my_thread_task(context, duration):
    context.set_message("Working in thread...")
    time.sleep(duration) # Simulate work
    context.set_progress(1.0)
    return "Thread task complete"

# Start the task in a thread
task_mgr.run_thread(my_thread_task, 2, key="thread_worker")
```

## आवश्यक पैटर्न

### UI अपडेट करना

बदलावों का प्रतिक्रिया देने के लिए `tasks_updated` सिग्नल से जुड़ें. हैंडलर सुरक्षित रूप से मुख्य
GTK थ्रेड पर कॉल किया जाएगा.

```python
def setup_ui(progress_bar, status_label):
    # This handler updates the UI based on the overall progress
    def on_tasks_updated(sender, tasks, progress):
        progress_bar.set_fraction(progress)
        if tasks:
            status_label.set_text(tasks[-1].get_message() or "Working...")
        else:
            status_label.set_text("Idle")

    task_mgr.tasks_updated.connect(on_tasks_updated)

# Later in your UI...
# setup_ui(my_progress_bar, my_label)
```

### रद्दीकरण

बाद में रद्द करने के लिए अपने कार्यों को एक `key` दें. आपका पृष्ठभूमि फ़ंक्शन समय-समय पर
`context.is_cancelled()` जाँचना चाहिए.

```python
# In your background function:
if context.is_cancelled():
    print("Task was cancelled, stopping work.")
    return

# In your UI code:
task_mgr.cancel_task("calculator")
```

### पूर्णता संभालना

परिणाम पाने या try/except के साथ त्रुटियाँ संभालने के लिए `when_done` कॉलबैक उपयोग करें:

```python
def on_task_finished(task):
    if task.get_status() == 'completed':
        try:
            result = task.result()
            print(f"Task finished with result: {result}")
        except Exception as e:
            print(f"Task failed: {e}")

task_mgr.run_process(my_cpu_task, 10, when_done=on_task_finished)
```

## API संदर्भ

### `task_mgr` (मैनेजर प्रॉक्सी)

- `add_coroutine(coro, *args, key=None, when_done=None)`: एक asyncio-आधारित कार्य जोड़ें
- `run_process(func, *args, key=None, when_done=None, when_event=None, visible=True)`: CPU-बाउंड
  कार्य अलग प्रक्रिया में चलाएँ. `visible` पैरामीटर नियंत्रित करता है कि कार्य प्रगति UI में दिखाई
  दे या नहीं. `when_event` कॉलबैक `context.send_event()` द्वारा भेजे गए कस्टम घटनाएँ प्राप्त करता
  है.
- `run_thread(func, *args, key=None, when_done=None)`: थ्रेड में कार्य चलाएँ (मुख्य प्रक्रिया के साथ
  मेमोरी साझा करता है)
- `cancel_task(key)`: कुंजी द्वारा चल रहा कार्य रद्द करें
- `tasks_updated` (UI अपडेट के लिए सिग्नल): कार्य स्थिति बदलने पर जारी होता है

### `context` (आपके पृष्ठभूमि फ़ंक्शन के भीतर)

- `set_progress(value)`: वर्तमान प्रगति रिपोर्ट करें (जैसे, `i + 1`)
- `set_total(total)`: `set_progress` के लिए अधिकतम मान सेट करें
- `set_message("...")`: स्थिति टेक्स्ट अपडेट करें
- `is_cancelled()`: जाँचें कि क्या आपको रुकना चाहिए
- `sub_context(...)`: बहु-चरण ऑपरेशनों के लिए एक उप-कार्य बनाएँ
- `send_event("name", data)`: (केवल प्रक्रिया) UI को वापस कस्टम डेटा भेजें
- `flush()`: कोई भी लंबित अपडेट तुरंत UI को भेजें

## Rayforge में उपयोग

टास्कर का उपयोग Rayforge भर में होता है:

- **पाइपलाइन प्रोसेसिंग**: दस्तावेज़ पाइपलाइन पृष्ठभूमि में चलाना
- **फ़ाइल ऑपरेशन**: UI अवरोधित किए बिना फ़ाइलें आयात और निर्यात करना
- **डिवाइस संचार**: लेज़र कटरों के साथ लंबे समय चलने वाले ऑपरेशन प्रबंधित करना
- **छवि प्रोसेसिंग**: CPU-गहन छवि ट्रेसिंग और प्रोसेसिंग करना

Rayforge में टास्कर के साथ काम करते समय, उत्तरदायी उपयोगकर्ता अनुभव बनाए रखने के लिए हमेशा सुनिश्चित
करें कि आपके पृष्ठभूमि फ़ंक्शन रद्दीकरण ठीक से संभालते हैं और सार्थक प्रगति अपडेट देते हैं.
