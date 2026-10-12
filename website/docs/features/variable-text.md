---
description:
  "Serial numbers and CSV merge in text templates: values that advance after every job, with
  consecutive values for array copies."
---

# Variable Text

Text boxes resolve `{...}` templates (see
[Parameters and Expressions](sketcher/expressions.md#template-expressions-in-text-boxes)). On top of
the sketch parameters and the built-in functions, the project offers variables that change from job
to job, so you can engrave numbered plates, tags and labels without editing the text each time.

## Serial numbers

Use `{serial}` in a text box. It is a whole number, so format specs control its appearance:

- `{serial}` — `7`
- `{serial:04d}` — `0007`, four digits with leading zeros
- `No. {serial:03d}` — `No. 007`

Open **Tools → Variable Text…** to set up the counter:

- **Current Value** — the number the next job starts with.
- **Start Value** — where **Reset** returns to.
- **Advance By** — the increment per copy and per job.
- **Wrap at End Value** / **End Value** — after the end value the counter continues at the start
  value.
- **Advance After Each Job** — when a job sent to the machine has finished, the counter moves on.
  Framing, pointer dry runs and cancelled jobs leave it alone.
- **Advance After G-code Export** — also move on when you export G-code, for jobs that run from a
  memory card.

The **Next Job** group previews the numbers the next job will use. **Previous** and **Next** move
the counter by one job, **Reset** returns to the start value. Every change, including the automatic
advance, is part of the undo history and is saved in the project.

## Numbering copies

Each workpiece has a **Variable Offset** (in the properties panel, under **Variable Text**). A copy
with offset _n_ shows the number _n_ copies after the current one. A
[grid or circular array](arrays.md) gives its copies increasing offsets automatically, so one array
of 20 plates engraves 20 consecutive numbers, and the next job continues with the 21st.

Arrays made inside a sketch are different: their copies belong to one workpiece and all show the
same number. Make the array on the canvas instead to number the copies.

For copies you placed yourself, select them all and choose **Arrange → Array → Number Copies**: the
offsets are assigned in reading order, rows from top to bottom and each row from left to right.

The counter advances by the number of copies in the job: the highest variable offset plus one.

## Merging a CSV file

For name tags, product labels and other lists, link a spreadsheet saved as CSV. Open **Tools →
Variable Text…** and choose the file under **CSV Merge**. The rows are stored in the project, so it
still works when the file is gone; **Reload** reads the file again after you changed it and keeps
the current row.

In a text box, refer to a column of the current row by name or by position, and to the row number:

- `{csv["Name"]}` — the `Name` column
- `{csv[0]}` — the first column
- `{row}` — the row number, starting at 1

The delimiter (comma, semicolon, tab or `|`) is detected and the file must be UTF-8. The first row
holds the column names, unless it consists only of numbers: then the columns are named `Column 1`,
`Column 2`, and so on, and can also be read by position.

Copies work like they do for serial numbers: a copy with variable offset _n_ shows the row _n_ rows
after the current one, so a grid array puts one row on each copy. **Copies Needed** tells you how
many copies cover the remaining rows in one job. Copies past the last row show empty columns.

With **Advance After Each Job** on, a finished job moves the current row on by the number of
copies, so for one row per job you just start the next job. After the last row it stops instead of
wrapping, and Rayforge tells you that every row has been used. **First**, **Previous** and **Next**
move the current row by hand. The serial number and the CSV row only advance when a text box uses
them.
