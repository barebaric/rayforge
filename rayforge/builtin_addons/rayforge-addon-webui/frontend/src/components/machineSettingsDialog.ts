import { api } from "../stores/api";
import { pushToast } from "../stores/uiStore";
import type {
  DriverInfo,
  DriverSetupPayload,
  MacroInfo,
  MachineSettings,
  NogoZoneInfo,
  SchemaGroup,
  WcsInfo,
} from "../api/types";
import {
  createSettingsForm,
  mergeFlatKeys,
  type SettingsField,
} from "./settingsDialog";
import {
  buildFormModel,
  type FormGroupModel,
} from "../steps/schemaForm";
import { buildField as buildSchemaField } from "./stepForm";
import { button } from "./ui/Button";
import { el } from "./ui/dom";
import { numberField, type NumberFieldHandle } from "./ui/NumberField";
import { select } from "./ui/Select";
import { tabs, type TabsHandle } from "./ui/Tabs";
import { textField } from "./ui/TextField";
import { toggle } from "./ui/Toggle";

const ORIGIN_OPTIONS = [
  { value: "bottom_left", label: "Bottom left" },
  { value: "bottom_right", label: "Bottom right" },
  { value: "top_left", label: "Top left" },
  { value: "top_right", label: "Top right" },
];

const GENERAL_FIELDS: SettingsField[] = [
  { key: "name", label: "Name", kind: "text" },
  { key: "origin", label: "Origin corner", kind: "choice", options: ORIGIN_OPTIONS },
  { key: "axis_extents.0", label: "Bed width", kind: "number", unit: "mm" },
  { key: "axis_extents.1", label: "Bed height", kind: "number", unit: "mm" },
  { key: "work_margins.0", label: "Margin left", kind: "number", unit: "mm" },
  { key: "work_margins.1", label: "Margin top", kind: "number", unit: "mm" },
  { key: "work_margins.2", label: "Margin right", kind: "number", unit: "mm" },
  { key: "work_margins.3", label: "Margin bottom", kind: "number", unit: "mm" },
  { key: "reverse_x_axis", label: "Reverse X", kind: "switch" },
  { key: "reverse_y_axis", label: "Reverse Y", kind: "switch" },
  { key: "reverse_z_axis", label: "Reverse Z", kind: "switch" },
  { key: "soft_limits_enabled", label: "Soft limits", kind: "switch" },
  { key: "max_travel_speed", label: "Max travel speed", kind: "number", unit: "mm/min" },
  { key: "max_cut_speed", label: "Max cut speed", kind: "number", unit: "mm/min" },
  { key: "acceleration", label: "Acceleration", kind: "number", unit: "mm/s²" },
  { key: "jog_speed", label: "Jog speed", kind: "number", unit: "mm/min" },
];

const HEAD_FIELDS: SettingsField[] = [
  { key: "max_power", label: "Max power (0-1000)", kind: "number" },
  { key: "frame_power", label: "Frame power (0-1)", kind: "number" },
  { key: "focus_power", label: "Focus power (0-1)", kind: "number" },
  { key: "frame_speed", label: "Frame speed (0 = machine default)", kind: "number", unit: "mm/min" },
  { key: "frame_repeat_count", label: "Frame repeat count", kind: "number" },
  { key: "frame_corner_pause", label: "Frame corner pause (s)", kind: "number" },
  { key: "spot_size_mm.0", label: "Spot width", kind: "number", unit: "mm" },
  { key: "spot_size_mm.1", label: "Spot height", kind: "number", unit: "mm" },
  { key: "focal_distance", label: "Focal distance (0 = auto)", kind: "number", unit: "mm" },
  { key: "pointer_offset_enabled", label: "Pointer offset", kind: "switch" },
  { key: "pointer_offset_x", label: "Pointer offset X", kind: "number", unit: "mm" },
  { key: "pointer_offset_y", label: "Pointer offset Y", kind: "number", unit: "mm" },
];

export interface MachineSettingsDialogDeps {
  machineId: string;
  /** Refetch viewport + machine state after settings changed. */
  onSaved?: () => void;
}

interface Loaded {
  settings: MachineSettings;
  driverSetup: DriverSetupPayload;
  drivers: DriverInfo[];
  macros: MacroInfo[];
}

function describe(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/**
 * Tabbed machine settings dialog in the style of the GTK device
 * profile editor: General geometry/motion, driver Device setup
 * (schema-driven VarSet form), laser heads, macros, work offsets,
 * no-go zones and rotary modules. Every tab saves through its own
 * endpoint; geometry changes re-push the viewport contract.
 */
export async function openMachineSettingsDialog(
  deps: MachineSettingsDialogDeps,
): Promise<void> {
  let loaded: Loaded;
  try {
    const [settings, driverSetup, drivers, macros] = await Promise.all([
      api.getMachineSettings(deps.machineId),
      api.getDriverSetup(deps.machineId),
      api.getDrivers().catch(() => [] as DriverInfo[]),
      api.getMacros(deps.machineId).catch(() => [] as MacroInfo[]),
    ]);
    loaded = { settings, driverSetup, drivers, macros };
  } catch (err) {
    pushToast("error", `Loading machine settings failed: ${describe(err)}`);
    return;
  }

  const body = el("div", "dialog-body settings-body machine-settings");
  const host = el("div", "machine-settings-content");
  body.appendChild(host);

  const backdrop = el("div", "modal-backdrop");
  const dialog = el("div", "dialog dialog-wide");
  const header = el("header", "dialog-header");
  const title = el("span", "dialog-title");
  title.textContent = `Machine settings — ${loaded.settings.name}`;
  header.append(
    title,
    button({ iconName: "x", variant: "icon", title: "Close", onClick: close }),
  );
  dialog.append(header, body);
  backdrop.appendChild(dialog);
  backdrop.addEventListener("click", (ev) => {
    if (ev.target === backdrop) close();
  });
  document.body.appendChild(backdrop);

  function close(): void {
    backdrop.remove();
  }

  function showSection(name: string): void {
    for (const [id, element] of sections) {
      element.hidden = id !== name;
    }
  }

  const sections = new Map<string, HTMLElement>();
  const tabDefs = [
    { id: "general", label: "General" },
    { id: "device", label: "Device" },
    { id: "laser", label: "Laser" },
    { id: "macros", label: "Macros" },
    { id: "offsets", label: "Work offsets" },
    { id: "nogo", label: "No-go zones" },
    { id: "rotary", label: "Rotary" },
  ];
  const tabHandle: TabsHandle = tabs(tabDefs, "general", (id) => {
    tabHandle.setActive(id);
    showSection(id);
  });
  body.insertBefore(tabHandle.root, host);

  sections.set("general", buildGeneralTab());
  sections.set("device", buildDeviceTab(loaded.driverSetup, loaded.drivers));
  sections.set("laser", buildLaserTab(loaded.settings));
  sections.set("macros", buildMacrosTab());
  sections.set("offsets", buildOffsetsTab(loaded.settings.coordinate_systems));
  sections.set("nogo", buildNogoTab(loaded.settings.nogo_zones));
  sections.set("rotary", buildRotaryTab(loaded.settings));
  for (const [, element] of sections) {
    host.appendChild(element);
  }
  showSection("general");

  function errorNoteInto(parent: HTMLElement): HTMLElement {
    const note = el("div", "panel-note settings-error");
    parent.appendChild(note);
    return note;
  }

  function buildGeneralTab(): HTMLElement {
    const root = el("div", "settings-section");
    const form = createSettingsForm(
      GENERAL_FIELDS,
      loaded.settings as unknown as Record<string, unknown>,
    );
    root.appendChild(form.root);
    root.appendChild(
      buildSaveRow(root, async () => {
        const changed = form.collect();
        if (Object.keys(changed).length === 0) return;
        await api.putMachineSettings(
          deps.machineId,
          mergeFlatKeys(
            loaded.settings as unknown as Record<string, unknown>,
            changed,
          ) as Parameters<typeof api.putMachineSettings>[1],
        );
      }),
    );
    return root;
  }

  function buildSaveRow(
    parent: HTMLElement,
    action: () => Promise<void>,
  ): HTMLElement {
    const row = el("div", "panel-actions");
    const note = errorNoteInto(parent);
    const saveBtn = button({
      label: "Save",
      iconName: "save",
      variant: "primary",
      onClick: () => {
        saveBtn.disabled = true;
        action()
          .then(() => {
            close();
            pushToast("success", "Machine settings saved");
            deps.onSaved?.();
          })
          .catch((err: unknown) => {
            note.textContent = describe(err);
            saveBtn.disabled = false;
          });
      },
    });
    row.appendChild(saveBtn);
    return row;
  }

  function buildDeviceTab(
    setup: DriverSetupPayload,
    drivers: DriverInfo[],
  ): HTMLElement {
    const root = el("div", "settings-section");
    const state: Record<string, unknown> = { ...setup.values };
    const dirty = new Set<string>();
    let pendingDriver = setup.driver;

    const driverRow = el("div", "schema-field-row");
    const driverLabel = el("span", "num-field-label");
    driverLabel.textContent = "Driver";
    const driverSelect = select(
      "Driver",
      drivers.map((entry) => ({
        value: entry.name,
        label:
          entry.name.replace(/Driver$/, "") +
          (entry.maturity_warning ? ` (${entry.maturity})` : ""),
      })),
      setup.driver,
      (value) => {
        pendingDriver = value;
      },
    );
    driverSelect.root.classList.add("in-panel");
    driverRow.append(driverLabel, driverSelect.root);
    root.appendChild(driverRow);

    const driverNote = el("div", "panel-note");
    driverNote.textContent =
      "Switching the driver resets its settings to the defaults.";
    root.appendChild(driverNote);

    if (setup.precheck_error) {
      const warn = el("div", "panel-note settings-error");
      warn.textContent = `Precheck: ${setup.precheck_error}`;
      root.appendChild(warn);
    }

    const groups: FormGroupModel[] = buildFormModel(
      { step_type: "driver", groups: [setup.setup] },
      setup.values,
    );
    const formWrap = el("div", "schema-group");
    const heading = el("div", "schema-group-title");
    heading.textContent = "Connection settings";
    formWrap.appendChild(heading);
    for (const group of groups) {
      for (const field of group.fields) {
        formWrap.appendChild(
          buildSchemaField(field, (key, value) => {
            state[key] = value;
            dirty.add(key);
          }),
        );
      }
    }
    root.appendChild(formWrap);

    if (setup.device_settings && setup.device_settings.length > 0) {
      const deviceNote = el("div", "panel-note");
      deviceNote.textContent =
        "Device ($$) settings are read from the connected hardware; " +
        "editing them from the web UI is not available yet.";
      root.appendChild(deviceNote);
      renderReadOnlyGroups(root, setup.device_settings, setup.values);
    }

    const row = el("div", "panel-actions");
    const note = el("div", "panel-note settings-error");
    const saveBtn = button({
      label: "Save",
      iconName: "save",
      variant: "primary",
      onClick: () => {
        saveBtn.disabled = true;
        void (async () => {
          try {
            const changed: Record<string, unknown> = {};
            for (const key of dirty) changed[key] = state[key];
            if (pendingDriver !== setup.driver) {
              await api.putDriver(deps.machineId, pendingDriver);
            }
            if (Object.keys(changed).length > 0) {
              await api.putDriverSetup(deps.machineId, changed);
            }
            close();
            pushToast("success", "Device settings saved");
            deps.onSaved?.();
          } catch (err) {
            note.textContent = describe(err);
            saveBtn.disabled = false;
          }
        })();
      },
    });
    row.appendChild(saveBtn);
    root.appendChild(row);
    return root;
  }

  function buildLaserTab(settings: MachineSettings): HTMLElement {
    const root = el("div", "settings-section");
    const headForms: Array<{
      uid: string;
      collect: () => Record<string, unknown>;
    }> = [];
    for (const head of settings.heads) {
      const group = el("div", "schema-group");
      const heading = el("div", "schema-group-title");
      heading.textContent = head.name || head.type;
      group.appendChild(heading);
      const values: Record<string, unknown> = { ...head } as Record<
        string,
        unknown
      >;
      delete values.uid;
      delete values.name;
      delete values.type;
      const form = createSettingsForm(HEAD_FIELDS, values);
      headForms.push({
        uid: head.uid,
        collect: () => mergeFlatKeys(values, form.collect()),
      });
      group.appendChild(form.root);
      root.appendChild(group);
    }
    if (settings.heads.length === 0) {
      const note = el("div", "panel-note");
      note.textContent = "This machine has no laser heads configured.";
      root.appendChild(note);
    }
    root.appendChild(
      buildSaveRow(root, async () => {
        await api.putHeads(
          deps.machineId,
          headForms.map((entry) => ({
            uid: entry.uid,
            ...entry.collect(),
          })) as Parameters<typeof api.putHeads>[1],
        );
      }),
    );
    return root;
  }

  function buildMacrosTab(): HTMLElement {
    const root = el("div", "settings-section");
    let macros = [...loaded.macros];

    const list = el("div", "macro-list");
    root.appendChild(list);

    const note = el("div", "panel-note settings-error");
    const refresh = () => {
      for (const child of [...list.children]) child.remove();
      if (macros.length === 0) {
        const empty = el("div", "panel-note");
        empty.textContent = "No macros defined.";
        list.appendChild(empty);
      }
      for (const macro of macros) {
        list.appendChild(macroRow(macro));
      }
    };

    const addBtn = button({
      label: "Add macro",
      iconName: "filePlus",
      onClick: () => {
        api
          .createMacro(deps.machineId, {
            name: `Macro ${macros.length + 1}`,
            code: "",
            enabled: true,
          })
          .then((created) => {
            macros.push(created);
            refresh();
          })
          .catch((err: unknown) =>
            pushToast("error", `Adding macro failed: ${describe(err)}`),
          );
      },
    });
    const actions = el("div", "panel-actions");
    actions.append(addBtn, note);
    root.appendChild(actions);

    function macroRow(macro: MacroInfo): HTMLElement {
      const group = el("div", "schema-group macro-item");
      const nameField = textField({
        label: "Name",
        value: macro.name,
        onCommit: () => undefined,
      });
      const codeLabel = el("span", "num-field-label");
      codeLabel.textContent = "G-code";
      const area = el("textarea", "textarea-input") as HTMLTextAreaElement;
      area.value = macro.code.join("\n");
      area.spellcheck = false;
      area.rows = 4;
      const enabledRow = el("div", "schema-field-row");
      const enabledLabel = el("span", "num-field-label");
      enabledLabel.textContent = "Enabled";
      const enabledToggle = toggle(macro.enabled, () => undefined);
      enabledRow.append(enabledLabel, enabledToggle.root);

      const rowActions = el("div", "panel-actions");
      const err = el("div", "panel-note settings-error");
      const saveBtn = button({
        label: "Save",
        onClick: () => {
          saveBtn.disabled = true;
          api
            .updateMacro(deps.machineId, macro.uid, {
              name: readText(nameField),
              code: area.value,
              enabled: enabledToggle.root.classList.contains("on"),
            })
            .then((updated) => {
              Object.assign(macro, updated);
              pushToast("success", "Macro saved");
            })
            .catch((error: unknown) => {
              err.textContent = describe(error);
              saveBtn.disabled = false;
            });
        },
      });
      const deleteBtn = button({
        label: "Delete",
        variant: "danger",
        onClick: () => {
          api
            .deleteMacro(deps.machineId, macro.uid)
            .then(() => {
              macros = macros.filter((entry) => entry.uid !== macro.uid);
              group.remove();
            })
            .catch((error: unknown) => {
              err.textContent = describe(error);
            });
        },
      });
      rowActions.append(saveBtn, deleteBtn, err);

      group.append(nameField.root, codeLabel, area, enabledRow, rowActions);
      return group;
    }

    refresh();
    return root;
  }

  function buildOffsetsTab(systems: WcsInfo[]): HTMLElement {
    const root = el("div", "settings-section");
    const note = el("div", "panel-note");
    note.textContent =
      "Work offsets are sent to the machine when a job starts; " +
      "use Sync in the control panel to re-read them from the device.";
    root.appendChild(note);

    for (const cs of systems) {
      const group = el("div", "schema-group wcs-row");
      const heading = el("div", "schema-group-title");
      heading.textContent = cs.name;
      group.appendChild(heading);
      const labelField = textField({
        label: "Label",
        value: cs.label,
        onCommit: () => undefined,
      });
      const axes = ["X", "Y", "Z"].map((axis, index) => ({
        axis,
        handle: numberField({
          label: axis,
          value: cs.offset[index] ?? 0,
          onCommit: () => undefined,
        }),
      }));
      const offsetRow = el("div", "field-grid");
      for (const entry of axes) offsetRow.appendChild(entry.handle.root);
      group.append(labelField.root, offsetRow);
      const rowActions = el("div", "panel-actions");
      const err = el("div", "panel-note settings-error");
      const saveBtn = button({
        label: "Save",
        onClick: () => {
          saveBtn.disabled = true;
          api
            .putWcs(
              deps.machineId,
              cs.name,
              axes.map((entry) => readNumber(entry.handle, 0)),
              readText(labelField),
            )
            .then(() => {
              pushToast("success", `${cs.name} saved`);
            })
            .catch((error: unknown) => {
              err.textContent = describe(error);
              saveBtn.disabled = false;
            });
        },
      });
      rowActions.append(saveBtn, err);
      group.appendChild(rowActions);
      root.appendChild(group);
    }
    return root;
  }

  function buildNogoTab(zones: NogoZoneInfo[]): HTMLElement {
    const root = el("div", "settings-section");
    const list = el("div", "nogo-list");
    const render = () => {
      for (const child of [...list.children]) child.remove();
      if (zones.length === 0) {
        const empty = el("div", "panel-note");
        empty.textContent = "No no-go zones defined.";
        list.appendChild(empty);
      }
      for (const zone of zones) list.appendChild(zoneRow(zone));
    };
    const zoneRow = (zone: NogoZoneInfo): HTMLElement => {
      const group = el("div", "schema-group nogo-item");
      const heading = el("div", "schema-group-title");
      heading.textContent = zone.name || "No-Go Zone";
      group.appendChild(heading);
      const summary = el("div", "panel-note");
      summary.textContent =
        `${zone.shape} · ` +
        Object.entries(zone.params)
          .map(([key, value]) => `${key}=${value}`)
          .join(", ");
      group.appendChild(summary);
      const rowActions = el("div", "panel-actions");
      const err = el("div", "panel-note settings-error");
      rowActions.append(
        button({
          label: "Delete",
          variant: "danger",
          onClick: () => {
            api
              .deleteNogoZone(deps.machineId, zone.uid)
              .then(() => {
                const index = zones.indexOf(zone);
                if (index >= 0) zones.splice(index, 1);
                group.remove();
                deps.onSaved?.();
              })
              .catch((error: unknown) => {
                err.textContent = describe(error);
              });
          },
        }),
        err,
      );
      group.appendChild(rowActions);
      return group;
    };
    render();
    root.appendChild(list);

    const addHeading = el("div", "schema-group-title");
    addHeading.textContent = "Add zone";
    root.appendChild(addHeading);
    const nameField = textField({
      label: "Name",
      value: "No-Go Zone",
      onCommit: () => undefined,
    });
    const shapeSelect = select(
      "Shape",
      [
        { value: "rect", label: "Rectangle" },
        { value: "cylinder", label: "Cylinder" },
      ],
      "rect",
      () => undefined,
    );
    const paramInputs = new Map<string, NumberFieldHandle>();
    const paramDefs: Array<[string, string]> = [
      ["x", "X"],
      ["y", "Y"],
      ["w", "Width"],
      ["h", "Height"],
      ["radius", "Radius"],
    ];
    for (const [key, label] of paramDefs) {
      paramInputs.set(
        key,
        numberField({ label, value: key === "w" || key === "h" ? 20 : 10 }),
      );
    }
    const paramGrid = el("div", "field-grid");
    for (const [, handle] of paramInputs) paramGrid.appendChild(handle.root);
    root.append(nameField.root, shapeSelect.root, paramGrid);

    const rowActions = el("div", "panel-actions");
    const err = el("div", "panel-note settings-error");
    rowActions.append(
      button({
        label: "Add zone",
        iconName: "filePlus",
        onClick: () => {
          const shape = readSelect(shapeSelect);
          const params: Record<string, number> = {};
          const keys = shape === "cylinder" ? ["x", "y", "radius"] : ["x", "y", "w", "h"];
          for (const key of keys) {
            const handle = paramInputs.get(key);
            if (handle) params[key] = readNumber(handle, 0);
          }
          api
            .createNogoZone(deps.machineId, {
              name: readText(nameField),
              shape,
              params,
            })
            .then((created) => {
              zones.push(created);
              render();
              deps.onSaved?.();
            })
            .catch((error: unknown) => {
              err.textContent = describe(error);
            });
        },
      }),
      err,
    );
    root.appendChild(rowActions);
    return root;
  }

  function buildRotaryTab(settings: MachineSettings): HTMLElement {
    const root = el("div", "settings-section");
    if (settings.rotary_modules.length === 0) {
      const note = el("div", "panel-note");
      note.textContent =
        "No rotary modules configured. Rotary modules are created in " +
        "the desktop app; their diameter and feed settings can be " +
        "edited here once they exist.";
      root.appendChild(note);
    }
    for (const module of settings.rotary_modules) {
      const group = el("div", "schema-group");
      const heading = el("div", "schema-group-title");
      heading.textContent = `${module.name} (${module.axis})`;
      group.appendChild(heading);
      const nameField = textField({
        label: "Name",
        value: module.name,
        onCommit: () => undefined,
      });
      const diameter = numberField({
        label: "Default diameter",
        value: module.default_diameter,
        unit: "mm",
        onCommit: () => undefined,
      });
      const mmPerRotation = numberField({
        label: "Movement per rotation",
        value: module.mm_per_rotation,
        unit: "mm",
        onCommit: () => undefined,
      });
      const reverseRow = el("div", "schema-field-row");
      const reverseLabel = el("span", "num-field-label");
      reverseLabel.textContent = "Reverse";
      const reverseToggle = toggle(module.reverse_axis, () => undefined);
      reverseRow.append(reverseLabel, reverseToggle.root);

      const rowActions = el("div", "panel-actions");
      const err = el("div", "panel-note settings-error");
      const saveBtn = button({
        label: "Save",
        onClick: () => {
          saveBtn.disabled = true;
          api
            .putRotaryModule(deps.machineId, module.uid, {
              name: readText(nameField),
              default_diameter: readNumber(diameter, module.default_diameter),
              mm_per_rotation: readNumber(
                mmPerRotation,
                module.mm_per_rotation,
              ),
              reverse_axis: reverseToggle.root.classList.contains("on"),
            })
            .then(() => pushToast("success", "Rotary module saved"))
            .catch((error: unknown) => {
              err.textContent = describe(error);
              saveBtn.disabled = false;
            });
        },
      });
      rowActions.append(saveBtn, err);
      group.append(
        nameField.root,
        diameter.root,
        mmPerRotation.root,
        reverseRow,
        rowActions,
      );
      root.appendChild(group);
    }
    return root;
  }
}

function renderReadOnlyGroups(
  parent: HTMLElement,
  groups: SchemaGroup[],
  values: Record<string, unknown>,
): void {
  for (const group of buildFormModel(
    { step_type: "device", groups },
    values,
  )) {
    const wrap = el("div", "schema-group");
    const heading = el("div", "schema-group-title");
    heading.textContent = group.title || "Device settings";
    wrap.appendChild(heading);
    for (const field of group.fields) {
      const rendered = buildSchemaField(field, () => undefined);
      rendered
        .querySelectorAll("input, textarea, select, button")
        .forEach((node) => {
          (node as HTMLInputElement).disabled = true;
        });
      wrap.appendChild(rendered);
    }
    parent.appendChild(wrap);
  }
}

function readText(field: { root: HTMLElement }): string {
  const input = field.root.querySelector("input") as HTMLInputElement | null;
  return input ? input.value : "";
}

function readSelect(field: { root: HTMLElement }): string {
  const select = field.root.querySelector("select") as HTMLSelectElement | null;
  return select ? select.value : "";
}

function readNumber(handle: NumberFieldHandle, fallback: number): number {
  const input = handle.root.querySelector("input") as HTMLInputElement | null;
  const parsed = input ? Number.parseFloat(input.value) : Number.NaN;
  return Number.isNaN(parsed) ? fallback : parsed;
}
