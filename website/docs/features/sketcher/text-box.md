---
description:
  "Place engraved text, labels, and serial numbers on a sketch with the Rayforge text box tool."
---

# Text Box

The text box tool (`G+T`) places text on the sketch as editable geometry — engraved text, labels,
and serial numbers. Text boxes are fully parametric: the glyphs live inside a constrained frame, so
they re-solve whenever the frame is moved or dimensioned.

![A wordmark and a part label](/screenshots/addons-sketcher-tool-text-box.webp)

## Creating and Editing Text

1. Pick the text box tool from the pie menu, the **Sketch** menu, or press `G+T`.
2. Click where you want the text to start: a text box appears at the click point and the tool
   switches straight into editing.
3. Type the text — the box resizes itself to fit as you type.
4. Press `Enter` or `Escape` to finish editing.

To edit an existing text box, click inside it. Double-click selects a word, triple-click the whole
line, and text can be selected and replaced like in any text editor, including `Ctrl+C`/`Ctrl+V`,
undo/redo, and paste mid-edit.

## Font Properties

![The font properties panel](/screenshots/addons-sketcher-tool-text-box-font-properties.webp)

The **Font Properties** panel in the sidebar controls the appearance of the text box selected in the
canvas:

- **Font family** — choose from the installed system fonts.
- **Font size** — in points.
- **Bold** and **Italic** toggles.

## A Parametric Frame

A text box is not a raster image: its glyphs are real sketch geometry, laid out inside a frame
defined by an origin and width and height points. The frame is drawn dashed as construction
geometry, so it serves as a layout reference and never ends up in the toolpaths when the sketch is
manufactured. Like everything else in the sketcher the frame is constrained, so it can be
dimensioned like any other geometry — change the width constraint and the text re-solves to fill the
box.

Clicking inside a text box with the [fill tool](fill.md) toggles the fill of the text glyphs instead
of creating a region fill.

## Template Expressions

Text boxes accept **template expressions**: anything in curly braces is evaluated when the sketch
solves, so labels can display live values such as dimensions, dates, or unique serial numbers. See
[Template expressions in text boxes](expressions.md#template-expressions-in-text-boxes) for details
and the built-in functions.
