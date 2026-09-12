This is an addendum to the primary AGENTS.md located in the Rayforge project root and is specific to the ruidarpa driver.

## Ruida Driver (ruidarpa) Development
- The focus is development of the ruidarpa driver. Because of this it is not necessary to run a complete test suite. Only the tests directly related to the ruidarpa driver are needed.
- The Ruida driver (ruidarpa) in development is in rayforge/machine/driver/ruidarpa. All edits must be confined to rayforge/machine/driver/ruidarpa and its subdirectories with the exception of ruidarpa specific tests in tests/machine/driver/ruidarpa.
- The Ruida driver in rayforge/machine/driver/ruida is a prototype and is for reference only. Never modify files in the rayforge/machine/driver/ruida directory tree.
- The ruidarpa driver depends upon external/ruida-pa which is maintained elsewhere. Do not modify files in the external/ruida-pa directory tree.
- If changes outside ruidarpa are needed, write a prompt which can passed to the upstream maintainers for either Rayforge or ruida-pa. Save the prompt in `docs/prompts` using a simple but descriptive file name.
- Temporary files should be created in rayforge/machine/driver/ruidarpa/tmp.
- `commit.txt`: To be written only when the user requests it (e.g. "Write a new commit.txt"). This contains pre-composed change summary lines and placed in the project root. Multiple summary lines are used when more than one issue (i.e. new feature or problem fix) has been resolved in a single commit. NOTE: Each issue should have only one line. Change details are to be written to the testing log (below). Each summary line should be prefixed a word indicating the nature of the change. These are:
	- `version:` The version number has been bumped.
	- `feature:` A new feature has been added.
	- `fix:` A problem has been fixed.
- Testing logs are maintained in `docs/logs` and have names formatted as ruidarpa-`<__version__>-testing.md`. **IMPORTANT:** The version (`<__version__>`) is the ruida-pa version. These are updated with detailed Problem, Solution and Verification information sections (not a bullet list). The testing log for the current version is to be updated when `commit.txt` is written.
