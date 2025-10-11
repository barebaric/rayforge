# RayForge Documentation Improvement Plan

**Date:** October 3, 2025

## Executive Summary

This plan outlines strategic improvements to the RayForge documentation. The primary goal is to transform the current foundation (44 files, 25+ empty stubs) into comprehensive, user-friendly documentation that serves beginners through advanced users.

### Current State

**RayForge Documentation Status:**
- ✅ Well-structured navigation with MkDocs Material
- ✅ Strong foundation: Good index pages, getting-started journey
- ⚠️ **Critical gap**: 25+ stub files (0 lines) - empty placeholders
- ⚠️ Limited visual content (only 12 images)
- ⚠️ Missing conceptual/"Explainer" content
- ⚠️ No task-based "How-To" guides beyond quick-start

### Documentation Goals

**Target structure:**
- 100+ comprehensive documentation pages
- 4-tier organization (Getting Started, Guides, Concepts, Collections)
- Extensive visual library (150+ images/diagrams)
- Clear audience segmentation (beginners, intermediate, advanced)
- Strong thematic collections for deep-dive learning

---

## Priority 1: Fill Critical Content Gaps (New Users)

**Problem:** Empty files are linked throughout docs but contain no content, breaking user journeys.

### Features Section (Core Functionality)

**Empty files to complete:**
- `features/camera.md` - Camera integration walkthrough
- `features/holding-tabs.md` - Tab placement guide
- `features/multi-layer.md` - Multi-layer workflow
- `features/macros-hooks.md` - G-code automation
- `features/overscan-kerf.md` - Quality improvement techniques
- `features/operations/contour.md` - Contour cutting operation
- `features/operations/raster.md` - Raster engraving operation
- `features/operations/depth.md` - Depth engraving operation
- `features/operations/shrink-wrap.md` - Shrink wrap operation
- `features/operations/material-test-grid.md` - Material test grid generator
- `features/simulation-mode.md` - Execution simulation and visualization

**Content requirements:**
- Step-by-step workflows with screenshots
- Use cases and when to use each feature
- Settings explanations with visual examples
- Common mistakes and troubleshooting tips

### Machine Setup (Essential for First-Time Users)

**Empty files to complete:**
- `machine/profiles.md` - Creating/managing machine profiles
- `machine/device-config.md` - Device configuration details
- `machine/grbl-settings.md` - GRBL parameter reference
- `machine/multi-laser.md` - Multiple laser setup

**Content requirements:**
- Configuration wizards and dialogs
- Parameter tables with explanations
- Screenshots of every settings screen
- Example configurations for common machines

### Troubleshooting (User Retention)

**Empty files to complete:**
- `troubleshooting/connection.md` - Serial/network connection issues
- `troubleshooting/common.md` - FAQ-style problem/solution pairs
- `troubleshooting/performance.md` - Optimization tips
- `troubleshooting/snap-permissions.md` - Linux permissions guide

**Content requirements:**
- Problem → Diagnosis → Solution format
- Step-by-step troubleshooting procedures
- Command examples with expected output
- Platform-specific solutions (Linux vs Windows)

**Impact:** Prevents broken user journeys and dead-end links, establishes credibility

---

## Priority 2: Add Conceptual "Explainer" Content (Understanding)

**Create a new `concepts/` section with foundational knowledge:**

### New Conceptual Articles

1. **`concepts/index.md`** - Overview of conceptual learning resources

2. **`concepts/understanding-operations.md`**
   - What are operations and when to use each type
   - Contour vs Raster vs Depth explained visually
   - Decision tree for operation selection
   - Real-world use case examples

3. **`concepts/coordinates-and-origin.md`**
   - How RayForge handles coordinate systems
   - Job origin vs machine origin vs workpiece origin
   - Common coordinate-related mistakes
   - Visual diagrams showing different origin modes

4. **`concepts/gcode-basics.md`**
   - What is G-code and how RayForge generates it
   - Understanding the generated code structure
   - When and how to edit G-code manually
   - Common G-code commands reference

5. **`concepts/laser-safety.md`**
   - Essential safety practices
   - Material hazards (never cut PVC, acrylic fumes, etc.)
   - Fire prevention and emergency procedures
   - Proper ventilation and PPE requirements
   - Emergency stop procedures

6. **`concepts/power-vs-speed.md`**
   - Relationship between power, speed, and material interaction
   - Multi-pass strategies for thick materials
   - Visual diagrams showing effects on different materials
   - How to read burn marks and adjust settings

**Impact:** Reduces support burden, empowers users to troubleshoot independently, builds deep understanding

---

## Priority 3: Create Task-Based "How-To Guides" (Practical)

**Create new `guides/` directory with specific, goal-oriented tutorials:**

### Getting Work Done

1. **`guides/index.md`** - Overview of available guides

2. **`guides/calibrating-your-workspace.md`**
   - Setting up accurate dimensions
   - Testing dimensional accuracy
   - Correcting steps-per-mm settings

3. **`guides/using-camera-alignment.md`**
   - Step-by-step camera workflow
   - Camera calibration process
   - Alignment techniques for precision work

4. **`guides/creating-material-test-grid.md`**
   - Using the built-in Material Test Grid generator
   - Understanding speed/power matrices
   - Configuring test parameters (speed range, power range, grid size)
   - Using presets (Diode/CO2, Cut/Engrave)
   - Interpreting test results and recording settings
   - Risk-optimized execution order explained

5. **`guides/batch-production-workflow.md`**
   - Repeating jobs efficiently
   - Using jigs and fixtures
   - Production optimization tips

### Quality & Precision

6. **`guides/achieving-perfect-focus.md`**
   - Laser focus techniques
   - Focus testing procedures
   - Variable focus for different materials

7. **`guides/reducing-burn-marks.md`**
   - Practical overscan/air-assist tips
   - Speed and power optimization
   - Material-specific techniques

8. **`guides/cutting-thick-materials.md`**
   - Multi-pass depth cutting strategies
   - Power/speed/pass calculations
   - Preventing charring

### Advanced Workflows

9. **`guides/using-simulation-mode.md`**
   - Understanding the simulation overlay in 2D view
   - Speed visualization with color heatmap (blue=slow, red=fast)
   - Power visualization with transparency
   - Laser head position tracking
   - Playback controls and scrubbing
   - Validating execution order before running jobs
   - Using simulation for material test debugging

10. **`guides/combining-raster-and-cut.md`**
    - Engrave-then-cut workflow
    - Layer organization best practices
    - Avoiding workpiece movement

11. **`guides/using-macros-for-automation.md`**
    - Real-world macro examples
    - Variable substitution
    - Common automation patterns

12. **`guides/rotary-attachment-setup.md`** (if supported)
    - Rotary configuration
    - Cylindrical object workflows

**Impact:** Converts intermediate users to power users, showcases advanced features, demonstrates real-world value

---

## Priority 4: Dramatically Expand Visual Content (Comprehension)

**Current state:** 12 images
**Target:** 150+ images/diagrams

### Establish Naming Convention

Recommended naming convention:
- `Guide-[Topic]-[Element][Number].png` - Tutorial screenshots
- `Ref-[Feature]-[State].png` - Reference screenshots
- `Diag-[Concept].png` - Diagrams and illustrations
- `UI-[Window]-[Tab].png` - Interface documentation

### Screenshot Strategy

1. **Every Major UI State/Dialog:**
   - Every settings panel mentioned in docs
   - Every step in multi-step guides
   - Before/after comparisons for operations
   - Error states and troubleshooting scenarios

2. **Create Custom Diagrams:**
   - Workflow flowcharts (expand Mermaid usage)
   - Technical diagrams for concepts (power vs speed, coordinates)
   - Annotated screenshots highlighting specific UI elements
   - Process diagrams showing data flow

3. **Visual Examples:**
   - Sample outputs showing different operations
   - Common mistakes vs correct results
   - Material test results with settings
   - Quality comparison images
   - Material test grid examples with different parameters
   - Simulation mode screenshots showing speed heatmap and power visualization
   - Annotated simulation screenshots explaining execution order

### Screenshot Placeholder System

To enable automated screenshot generation and maintain consistency, use standardized HTML comment placeholders in documentation files:

**Simple placeholder:**
```markdown
```yaml
# <!-- SCREENSHOT: ui-settings-machine-profile
# description: Machine settings dialog showing Profiles tab
# filename: UI-Settings-MachineProfile.png
# -->
```

![Machine profile settings](../images/UI-Settings-MachineProfile.png)
```

**Detailed placeholder with automation steps:**
```markdown
```yaml
# <!-- SCREENSHOT
# id: guide-simulation-mode-heatmap
# type: screenshot
# description: Simulation mode with speed heatmap at 60% progress
# setup:
#   - action: open_example
#     name: material-test-grid
#   - action: press_key
#     key: F7
#   - action: play_simulation
#   - action: pause_at
#     progress: 0.6
# annotations:
#   - type: callout
#     text: "Speed heatmap: Blue=slow, Red=fast"
# filename: Guide-SimulationMode-Heatmap.png
# -->
```
```

See [markdown_screenshot_spec.md](markdown_screenshot_spec.md) for complete specification.

### Image Guidelines

- **Format:** PNG for screenshots, use mermaid markdown for diagrams where possible or svg as fallback
- **Resolution:** High-DPI screenshots (2x scale), then compress
- **Annotations:** Use consistent color scheme for callouts (#FF6B00 for highlights)
- **Alt text:** Descriptive alt text for accessibility
- **File size:** Optimize with compression tools
- **Placeholders:** Always include screenshot placeholders for automated regeneration

**Impact:** dramatically improves comprehension and reduces support questions

---

## Priority 5: Build Reference Section (Advanced Users)

**Currently sparse - expand with detailed technical content:**

### Complete Existing Empty Files

1. **`reference/shortcuts.md`**
   - Complete keyboard shortcut table
   - Categorized by function (File, Edit, View, Machine, etc.)
   - Searchable/filterable format
   - Platform-specific variations (Linux vs Windows)
   - Include F7 for Simulation Mode toggle
   - Include shortcuts for material test grid operations

2. **`reference/gcode-dialects.md`**
   - Detailed comparison of GRBL vs Smoothieware G-code
   - RayForge-specific commands and variables
   - Dialect-specific limitations
   - Compatibility matrix by firmware version

3. **`reference/firmware.md`**
   - Firmware compatibility matrix
   - Version-specific features/bugs
   - Upgrade recommendations
   - Known issues by version

### New Reference Content

4. **`reference/command-reference.md`**
   - All menu commands documented
   - Keyboard shortcuts listed
   - Command descriptions and use cases
   - CLI arguments if applicable

5. **`reference/file-format-spec.md`**
   - RayForge project file format documentation
   - Import/export capabilities by format
   - Compatibility notes with other software
   - Version differences

6. **`reference/settings-reference.md`**
   - Complete settings/preferences documentation
   - Every option explained
   - Default values and valid ranges
   - Impact on performance and quality

**Impact:** Power users and integrators need this; builds credibility, enables advanced usage

---

## Priority 6: Enhance "Getting Started" Journey (Onboarding)

**Current:** Good but can be broken into more granular, focused articles

### Break Down Quick Start into Focused Articles

Current state: `quick-start.md` (170 lines covering 7 steps)

**Proposed breakdown:**

1. **`getting-started/importing-your-first-file.md`**
   - File format selection
   - Import dialog walkthrough
   - Common import issues

2. **`getting-started/understanding-the-canvas.md`**
   - Canvas navigation
   - Zoom and pan controls
   - Object manipulation basics

3. **`getting-started/your-first-operation.md`**
   - Choosing the right operation
   - Basic operation settings
   - Simple contour cut example

4. **`getting-started/using-3d-preview.md`**
   - Preview navigation
   - What to look for in preview
   - Common preview issues

5. **`getting-started/simulating-your-job.md`**
   - Activating simulation mode (F7)
   - Understanding the speed heatmap (blue=slow, red=fast)
   - Using playback controls and scrubbing
   - Validating execution order before running
   - What to check for in the simulation

6. **`getting-started/framing-your-job.md`**
   - Framing purpose and process
   - Positioning material
   - Adjusting job placement

7. **`getting-started/running-your-first-job.md`**
   - Pre-flight checklist
   - Starting the job
   - Monitoring progress
   - Emergency stop

8. **`getting-started/next-steps.md`**
   - What to learn next
   - Recommended learning path
   - Links to guides and concepts

### Add Progressive Complexity Indicators

- **Level 1:** Absolute basics (getting-started)
- **Level 2:** Intermediate skills (guides)
- **Level 3:** Advanced techniques (advanced guides)

**Impact:** Reduces overwhelm, increases completion rate for new users, improves retention

---

## Priority 7: Add Collection/Topic Pages (Discovery)

**Create thematic collections to organize related content:**

### New Collections Section

1. **`collections/index.md`** - Overview of available collections

2. **`collections/new-user-essentials.md`**
   - Curated links to must-read content for beginners
   - Ordered learning path with checkboxes
   - Estimated time to complete
   - Prerequisites for each section

3. **`collections/workflow-optimization.md`**
   - All guides related to efficiency and quality
   - Tips and tricks compilation
   - Pro user techniques
   - Time-saving shortcuts

4. **`collections/troubleshooting-guide.md`**
   - Problem-solution pairs by category
   - Decision trees for diagnosis
   - Common error messages explained
   - When to seek help

5. **`collections/material-library.md`**
   - Community-contributed material settings
   - Material safety information
   - Best practices by material type
   - Links to external resources

**Impact:** Helps users discover related content, improves navigation, creates curated learning paths

---

## Priority 8: File Formats Section (Currently Minimal)

**Complete empty files with comprehensive content:**

### Files to Complete

1. **`files/importing.md`**
   - Detailed import workflows for each format
   - Format-specific settings and options
   - Import dialog screenshots for each format
   - Common import problems and solutions

2. **`files/formats.md`**
   - Format capabilities comparison table
   - Supported features by format
   - Recommended formats for different use cases
   - Format conversion tips

3. **`files/exporting.md`**
   - Export options and settings
   - G-code export customization
   - Exporting for different firmware types
   - Post-processing options

### Additional Content

- Format-specific import tips (SVG optimization, DXF units, PDF layer handling)
- File preparation in design tools (Inkscape, Illustrator, etc.)
- Troubleshooting malformed files
- Best practices for file organization

**Impact:** Reduces import friction, helps users prepare better source files

---

## Priority 9: Add Downloadable Resources

**Provide downloadable resources for users:**

### Create Downloads Directory

**New directory:** `docs-site/assets/downloads/`

### Sample Files

1. **`sample-project.svg`** - Basic shapes for first test
   - Simple square, circle, text
   - Pre-sized for common laser beds
   - Demonstration of different object types

2. **`calibration-grid.svg`** - Dimensional accuracy test
   - Measured grid pattern
   - Reference dimensions
   - Instructions in file metadata

3. **`example-macro-library.txt`** - Common G-code macros
   - Well-commented examples
   - Usage instructions
   - Variable substitution examples

4. **`troubleshooting-checklist.pdf`** - Printable reference
   - Quick diagnosis flowchart
   - Common fixes
   - When to seek help

### Integration

- Reference downloads in relevant guides
- Link from getting-started for immediate hands-on experience
- Include download counts/usage stats if possible

**Impact:** Reduces friction, gets users producing results faster, demonstrates best practices

---

## Implementation Strategy

### Phase 1: Critical Path (Weeks 1-2)

**Goal:** Remove all broken links and complete core user journeys

- [x] Document Material Test Grid Generator (`features/operations/material-test-grid.md`)
- [x] Document Simulation Mode (`features/simulation-mode.md`)
- [x] Fill all empty operation files (contour, raster, depth, shrink-wrap)
- [x] Complete machine setup section (profiles, device-config, grbl-settings)
- [x] Finish core troubleshooting (connection, common problems, performance, snap-permissions)
- [x] Complete all empty feature files (camera, holding-tabs, macros-hooks, multi-layer, overscan-kerf)
- [ ] Screenshot every UI element mentioned in existing docs
- [ ] Screenshot Material Test Grid settings dialog and examples
- [ ] Screenshot Simulation Mode with speed heatmap and playback controls
- [x] Create `concepts/laser-safety.md` (critical safety content)

**Deliverable:** All linked pages have content, no dead ends ✅ COMPLETE

### Phase 2: Foundation (Weeks 3-4)

**Goal:** Build conceptual understanding

- [ ] Create `concepts/` section with index
- [ ] Write 5 key conceptual articles (operations, coordinates, power-speed, gcode, safety)
- [x] Expand reference section (shortcuts, g-code, firmware)
- [ ] Add 30+ screenshots for new content
- [ ] Create first set of diagrams (10+ custom illustrations)

**Deliverable:** Users can understand "why" not just "how" - IN PROGRESS

### Phase 3: Enrichment (Weeks 5-6)

**Goal:** Provide practical, task-based guidance

- [ ] Build `guides/` section with index
- [ ] Write `guides/creating-material-test-grid.md`
- [ ] Write `guides/using-simulation-mode.md`
- [ ] Write top 6-8 additional how-to articles (calibration, camera, etc.)
- [ ] Create downloadable sample files (3-4 essential files)
- [ ] Add 50+ workflow screenshots
- [ ] Complete file formats section

**Deliverable:** Users can accomplish real tasks independently

### Phase 4: Polish (Week 7+)

**Goal:** Enhance discoverability and create learning paths

- [ ] Create `collections/` section with curated content
- [ ] Add remaining how-to guides (advanced topics)
- [ ] Expand troubleshooting with more problem/solution articles
- [ ] Reach 150+ total images
- [ ] Gather user feedback and iterate

**Deliverable:** Comprehensive, discoverable documentation

### Phase 5: Advanced (Ongoing)

**Optional enhancements:**

- [ ] Video tutorials for complex workflows
- [ ] Interactive diagrams (if feasible with MkDocs)
- [ ] Community contributions section
- [ ] Translation improvements
- [ ] Search optimization

---

## Proposed Navigation Structure

**Reorganized `mkdocs.yml` nav section:**

```yaml
nav:
  - Home: index.md

  - Getting Started:
      - getting-started/index.md
      - Installation: getting-started/installation.md
      - First Time Setup: getting-started/first-time-setup.md
      - Importing Your First File: getting-started/importing-your-first-file.md
      - Understanding the Canvas: getting-started/understanding-the-canvas.md
      - Your First Operation: getting-started/your-first-operation.md
      - Using 3D Preview: getting-started/using-3d-preview.md
      - Simulating Your Job: getting-started/simulating-your-job.md
      - Framing Your Job: getting-started/framing-your-job.md
      - Running Your First Job: getting-started/running-your-first-job.md
      - Next Steps: getting-started/next-steps.md

  - Concepts:
      - concepts/index.md
      - Understanding Operations: concepts/understanding-operations.md
      - Coordinates & Origin: concepts/coordinates-and-origin.md
      - Power vs Speed: concepts/power-vs-speed.md
      - G-code Basics: concepts/gcode-basics.md
      - Laser Safety: concepts/laser-safety.md

  - How-To Guides:
      - guides/index.md
      - Material Test Grid: guides/creating-material-test-grid.md
      - Simulation Mode: guides/using-simulation-mode.md
      - Calibrating Your Workspace: guides/calibrating-your-workspace.md
      - Camera Alignment: guides/using-camera-alignment.md
      - Batch Production: guides/batch-production-workflow.md
      - Achieving Perfect Focus: guides/achieving-perfect-focus.md
      - Reducing Burn Marks: guides/reducing-burn-marks.md
      - Cutting Thick Materials: guides/cutting-thick-materials.md
      - Combining Raster and Cut: guides/combining-raster-and-cut.md
      - Using Macros: guides/using-macros-for-automation.md

  - User Interface:
      - ui/index.md
      - Main Window: ui/main-window.md
      - Canvas Tools: ui/canvas-tools.md
      - 3D Preview: ui/3d-preview.md
      - Settings & Preferences: ui/settings.md

  - Features:
      - features/index.md
      - Simulation Mode: features/simulation-mode.md
      - Operations:
          - features/operations/index.md
          - Material Test Grid: features/operations/material-test-grid.md
          - Contour Cutting: features/operations/contour.md
          - Raster Engraving: features/operations/raster.md
          - Depth Engraving: features/operations/depth.md
          - Shrink Wrap: features/operations/shrink-wrap.md
      - Multi-Layer Workflow: features/multi-layer.md
      - Camera Integration: features/camera.md
      - Holding Tabs: features/holding-tabs.md
      - G-code Macros & Hooks: features/macros-hooks.md
      - Overscan & Kerf: features/overscan-kerf.md

  - Machine Setup:
      - machine/index.md
      - Machine Profiles: machine/profiles.md
      - Device Configuration: machine/device-config.md
      - GRBL Settings: machine/grbl-settings.md
      - Multiple Lasers: machine/multi-laser.md

  - File Formats:
      - files/index.md
      - Importing Files: files/importing.md
      - Supported Formats: files/formats.md
      - Exporting G-code: files/exporting.md

  - Troubleshooting:
      - troubleshooting/index.md
      - Connection Issues: troubleshooting/connection.md
      - Common Problems: troubleshooting/common.md
      - Performance: troubleshooting/performance.md
      - Permissions (Snap): troubleshooting/snap-permissions.md

  - Reference:
      - reference/index.md
      - Keyboard Shortcuts: reference/shortcuts.md
      - Command Reference: reference/command-reference.md
      - G-code Dialect Support: reference/gcode-dialects.md
      - Firmware Compatibility: reference/firmware.md
      - Settings Reference: reference/settings-reference.md
      - File Format Specifications: reference/file-format-spec.md

  - Collections:
      - collections/index.md
      - New User Essentials: collections/new-user-essentials.md
      - Workflow Optimization: collections/workflow-optimization.md
      - Troubleshooting Guide: collections/troubleshooting-guide.md
      - Material Library: collections/material-library.md

  - Contributing:
      - contributing/index.md
      - Development Setup: contributing/development.md
      - Adding Device Drivers: contributing/drivers.md
```

---

## Success Metrics

### Quantitative Goals

- **Content volume:** 44 → 100+ substantive pages (from 25 empty stubs)
- **Visual aids:** 12 → 150+ images/diagrams
- **Complete user journeys:** 0 broken links or dead ends
- **Reference coverage:** 100% of features documented

### Qualitative Goals

- **New user success rate:** Significant improvement in onboarding completion
- **Support burden:** Reduced through self-service troubleshooting and concepts
- **User satisfaction:** Positive feedback on documentation completeness
- **Discoverability:** Users can find answers without external help

### Tracking

- Monitor GitHub issues related to documentation gaps
- Track common support questions that should be documented
- User feedback on documentation helpfulness
- Analytics on most-visited pages (identify gaps)

---

## Content Creation Guidelines

### Writing Style

- **Audience:** End-users, not developers
- **Tone:** Friendly, clear, and concise
- **Voice:** Use second person ("you") and active voice
- **Examples:** Provide concrete examples and screenshots
- **Safety:** Always emphasize safety when relevant

### Page Structure

Each page should include:

1. **Title:** Clear, descriptive H1 heading
2. **Introduction:** Brief overview (2-3 sentences)
3. **Prerequisites:** What users need before starting (if applicable)
4. **Body:** Detailed information with examples, organized with H2/H3 headings
5. **Tips/Warnings:** Callout boxes for important information
6. **Navigation:** "Next steps" or "Related pages" links at the end

### Visual Content Standards

- **Screenshots:**
  - Use default theme for consistency
  - Crop to relevant area, but include enough context
  - Annotate with arrows/highlights if needed
  - Include alt text describing what's shown

- **Diagrams:**
  - Use Mermaid for flowcharts when possible
  - Create custom SVG diagrams for complex concepts
  - Use consistent color scheme and style
  - Include legends for symbols

- **Code Examples:**
  - Syntax highlighted
  - Include comments explaining non-obvious parts
  - Show expected output when relevant
  - Test all examples before publishing

---

## RayForge Documentation Focus

The RayForge documentation should emphasize:

1. **Open-source collaboration:**
   - Contributing guide is prominent
   - Community involvement encouraged
   - Development transparency
   - "Edit this page" links enabled for community contributions

2. **Modern technology:**
   - Highlight modern UI (Gtk4/Libadwaita)
   - Cross-platform capabilities
   - Integration possibilities

3. **Target audience:**
   - Hobbyists and makers (accessible pricing/free)
   - Educational use
   - Small businesses

4. **Unique features:**
   - Multi-laser support
   - Advanced G-code preview
   - Extensibility and customization
   - Built-in Material Test Grid Generator (parametric, risk-optimized)
   - Real-time Simulation Mode (speed heatmap, power visualization)

---

## Resources Needed

### Content Creation

- **Technical writer:** 40-80 hours for Phase 1-2
- **Subject matter expert:** Review and accuracy checking
- **Graphic designer:** Custom diagrams and illustrations (optional)

### Tools

- **Screenshot tools:** Built-in OS tools sufficient
- **Image editing:** GIMP or similar (annotations, cropping)
- **Diagram tools:** Mermaid (built-in), draw.io, or Inkscape for complex diagrams
- **Screen recording:** OBS Studio for video tutorials (Phase 5)

### Review Process

1. **Technical accuracy:** SME review of technical content
2. **User testing:** Beta testers validate guides with fresh eyes
3. **Editorial review:** Grammar, style, consistency
4. **Accessibility:** Alt text, screen reader compatibility
5. **Cross-linking:** Verify all internal links work

---

## Maintenance Plan

### Ongoing Updates

- **Release notes:** Document new features in relevant sections
- **Deprecation notices:** Mark outdated content clearly
- **Version-specific content:** Use tabs for version differences
- **Link checking:** Automated link checking in CI/CD

### Community Contributions

**Enable "Edit this page" feature:**

Already configured in `mkdocs.yml`:

This allows users to click "Edit this page" on any documentation page and be taken directly to the GitHub editor for that file, making it easy to submit corrections and improvements via pull requests.

**Workflow:**
- **Contribution guidelines:** Clear process for doc contributions
- **Recognition:** Credit contributors in changelog
- **Moderation:** Review process for community PRs
- **Easy access:** Users can click "Edit this page" to submit improvements directly

### Feedback Loop

- **Feedback widget:** "Was this helpful?" on each page
- **Issue tracking:** Document requests in GitHub issues
- **Analytics:** Monitor popular pages and search terms
- **User surveys:** Periodic documentation satisfaction surveys

---

## Conclusion

This plan transforms RayForge documentation from a solid foundation with gaps into comprehensive, user-friendly documentation that serves users from first install through advanced usage. By following LightBurn's proven structure while maintaining RayForge's unique voice and community focus, we can create documentation that:

1. **Reduces support burden** through comprehensive self-service content
2. **Improves user success** with clear onboarding and task-based guides
3. **Builds user confidence** through conceptual understanding
4. **Serves all skill levels** from beginners to advanced users
5. **Enhances project credibility** with professional, complete documentation

The phased approach ensures critical gaps are filled first, with progressive enhancement adding depth and discoverability over time.

---

## Next Steps

1. **Review and approve** this plan with stakeholders
2. **Prioritize** specific articles within each phase based on user feedback
3. **Assign resources** for content creation and review
4. **Set up tracking** for metrics and progress
5. **Begin Phase 1** with critical content gaps

**Estimated timeline:** 7-10 weeks for Phases 1-4, with ongoing maintenance and enhancement.
