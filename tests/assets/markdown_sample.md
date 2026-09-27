# Markdown Parser Test Document

This document exercises every construct of the limited Markdown subset that
Rayforge supports. Paste it into **Machine Settings -> Notes -> Edit My Notes**
to check the renderer by hand. It is also parsed by
`tests/shared/markdown/test_sample_document.py`, which fails if a supported
construct stops being recognised.

The "Safety And Malformed Input" section near the end deliberately contains
raw HTML, an unsafe link and unbalanced markup. That content is meant to stay
literal, so please do not "fix" it.

This is a standard paragraph to test how the parser handles regular body text. In this section, we also combine **bold text** and *italicized text* within the same paragraph.

## Features & Specifications

Here is an unordered list to test list formatting:

* **Item 1**: First item containing an **important** note.

* **Item 2**: Second item where we include a link to [Google](https://www.google.com?utm_source=gemini).

* **Item 3**: Third item demonstrating `inline code`.

Note: the blank lines above split these into three separate single-item
lists. The tight list below is parsed as one list of three items, and may
use any of the three bullet markers:

- Dash bullet with ***bold italic text***.
+ Plus bullet with __underscore bold__ and _underscore italic_.
- Dash bullet with ___underscore bold italic___.

1. first
2. second
3. third

### Heading Levels

#### Level 4 Heading

##### Level 5 Heading

###### Level 6 Heading

### Quotes And Line Breaks

> Always wear laser safety glasses.
> This quote spans two source lines.

A paragraph can also contain a soft line break.
This sentence started on the next source line.

> A quote may contain **bold text**, a [link](https://rayforge.org) and
> `inline code` as well.

### Code Block Testing

Below is a standard fenced code block written in JavaScript:

```
function greet(name) {
    console.log(`Hello, ${name}! Welcome to the Markdown test.`);
}

greet("Tester");

```

### Custom Expandable Details Test

Here we test the custom expandable details syntax containing a multi-line code snippet inside:

:::details Click here to expand the hidden content
Here is some introductory text inside the expandable section.

```python
def calculate_sum(numbers):
    total = 0
    for number in numbers:
        total += number
    return total


print(calculate_sum([10, 20, 30]))
```

You can also include **bold text** or a [helpful link](https://markdown-guide.org?utm_source=gemini) inside this section.
:::enddetails

### Repeated Details Titles

Both sections below share a title. Expand only the second one, then keep
typing in the editor: it should stay open while the first stays closed.

:::details Shared title
Content of the first section.

- A list inside a details section.
- A second entry.

> And a quote inside a details section.
:::enddetails

:::details Shared title
Content of the second section.
:::enddetails

### Safety And Malformed Input

None of the following may render as markup or become clickable:

Raw HTML stays literal: <script>alert('test')</script> and <b>not bold</b>.

An unsafe link must render as plain text: [do not click](javascript:alert('test')).

An unclosed details block stays literal text:

:::details This block is never closed

Unbalanced emphasis also stays literal: **only one pair of stars.

*Good luck testing markdown rendering!*
