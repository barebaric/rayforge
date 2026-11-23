---
description:
  "A job counter in text templates and G-code: one project-level number that advances after every
  job, with consecutive numbers for array copies."
---

# Job Counter

Text boxes resolve `{...}` templates (see
[Parameters and Expressions](sketcher/expressions.md#template-expressions-in-text-boxes)). On top of
the sketch parameters and the built-in functions, the project has one job counter that changes from
job to job, so you can engrave numbered plates, tags and labels without editing the text each time.

## The job counter

Use `{counter}` in a text box. It is a whole number, so format specs control its appearance:

- `{counter}` — `7`
- `{counter:04d}` — `0007`, four digits with leading zeros
- `No. {counter:03d}` — `No. 007`

Open **Tools → Job Counter…** to set up the counter:

- **Current Value** — the number the next job starts with.
- **Start Value** — where **Reset** returns to.
- **Advance By** — the increment per copy and per job.
- **Wrap at End Value** / **End Value** — after the end value the counter continues at the start
  value.
- **Advance After Each Job** — when a job sent to the machine has finished, the counter moves on.
  Framing, pointer dry runs, cancelled jobs and failed jobs leave it alone.
- **Advance After G-code Export** — also move on when you export G-code, for jobs that run from a
  memory card.

The **Next Job** group previews the numbers the next job will use. **Previous** and **Next** move
the counter by one job, **Reset** returns to the start value. Every change, including the automatic
advance, is part of the undo history and is saved in the project.

## The counter in G-code

The counter is also available to G-code hooks, macros and the dialect's preamble and postscript:

- `{counter}` is the value the job starts with.
- `{workpiece.counter}` is the value for the current workpiece, so a **Workpiece Start** hook can
  label each copy individually.
- Format specs work here too: `{counter:04d}`.

For example, a preamble line `; Job {counter:04d}` prints the starting number, and a Workpiece Start
hook containing `; Part {workpiece.counter:04d}` prints each part's number. See
[Macros & Hooks](../machine/hooks-macros.md#variable-substitution) for the other variables.

## Numbering copies

Each workpiece has an offset. A copy with offset _n_ shows the number _n_ copies after the current
one. A [grid or circular array](arrays.md) and **Arrange → Array → Number Copies** give the copies
increasing offsets, so one array of 20 plates engraves 20 consecutive numbers, and the next job
continues with the 21st.

Arrays made inside a sketch are different: their copies belong to one workpiece and all show the
same number. Make the array on the canvas instead to number the copies.

A new copy of a sketch that reads the counter is numbered automatically: the next instance gets the
next offset, so placing two instances of the same numbered sketch shows consecutive values. To
renumber copies you placed earlier, select them all and choose **Arrange → Array → Number Copies**:
the offsets are assigned in reading order, rows from top to bottom and each row from left to right.

The counter advances by the number of copies in the job: the highest variable offset plus one. If
the counter is only used in G-code templates, every workpiece in the job counts.
