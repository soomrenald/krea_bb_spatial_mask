import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const LOADER_NODE = "Krea2MultiLoRALoader";
const JSON_WIDGET = "loras_json";
let LORA_LIST = ["None"];
const DEFAULT_COLORS = ["#ff5f57", "#febc2e", "#28c840", "#5fb3ff", "#af7cff", "#ff75b5"];

async function ensureLoras() {
  if (LORA_LIST.length > 1) return LORA_LIST;
  try {
    const response = await api.fetchApi("/object_info/LoraLoader");
    const info = await response.json();
    const names = info?.LoraLoader?.input?.required?.lora_name?.[0];
    if (Array.isArray(names) && names.length) {
      LORA_LIST = ["None", ...names.filter((n) => n !== "None")];
    }
  } catch (err) {
    console.warn("[Krea2MultiLoRALoader] Could not fetch LoRA list", err);
  }
  return LORA_LIST;
}

function jsonWidget(node) {
  return node.widgets?.find((w) => w.name === JSON_WIDGET);
}

function readSelections(node) {
  const w = jsonWidget(node);
  if (!w) return [];
  try {
    const parsed = JSON.parse(w.value || "[]");
    return Array.isArray(parsed) ? parsed : [];
  } catch (_) {
    return [];
  }
}

function writeSelections(node, selections) {
  const w = jsonWidget(node);
  if (!w) return;
  w.value = JSON.stringify(selections, null, 2);
  if (w.inputEl) w.inputEl.value = w.value;
  node.setDirtyCanvas(true, true);
}

function markTransient(widget) {
  widget.__k2_loader_widget = true;
  widget.serialize = false;
  widget.options = widget.options || {};
  widget.options.serialize = false;
  return widget;
}

function defaultSelection(index) {
  return {
    enabled: true,
    alias: `character_${index + 1}`,
    lora: "None",
    strength: 1.0,
    boxes: String(index + 1),
    color: DEFAULT_COLORS[index % DEFAULT_COLORS.length],
  };
}

function updateSelection(node, idx, updater) {
  const selections = readSelections(node);
  if (!selections[idx]) return;
  updater(selections[idx]);
  writeSelections(node, selections);
}

function rebuildRows(node) {
  if (!node.widgets) return;
  node.widgets = node.widgets.filter((w) => !w.__k2_loader_widget);
  const selections = readSelections(node);

  selections.forEach((sel, idx) => {
    const header = node.addWidget("text", `LoRA ${idx + 1} label`, `${idx + 1}: ${sel.alias || `character_${idx + 1}`}`, () => {});
    header.disabled = true;
    markTransient(header);

    const enableWidget = node.addWidget(
      "toggle",
      `LoRA ${idx + 1} enabled`,
      sel.enabled !== false,
      (value) => updateSelection(node, idx, (s) => { s.enabled = value; }),
      { on: "on", off: "off" }
    );
    markTransient(enableWidget);

    const aliasWidget = node.addWidget("text", `LoRA ${idx + 1} alias`, sel.alias || `character_${idx + 1}`, (value) => {
      updateSelection(node, idx, (s) => { s.alias = value; });
      setTimeout(() => rebuildRows(node), 0);
    });
    markTransient(aliasWidget);

    const loraWidget = node.addWidget(
      "combo",
      `LoRA ${idx + 1} file`,
      sel.lora || "None",
      (value) => updateSelection(node, idx, (s) => { s.lora = value; }),
      { values: LORA_LIST }
    );
    markTransient(loraWidget);

    const strengthWidget = node.addWidget(
      "number",
      `LoRA ${idx + 1} strength`,
      Number.isFinite(sel.strength) ? sel.strength : 1.0,
      (value) => updateSelection(node, idx, (s) => { s.strength = Number(value); }),
      { min: -5, max: 5, step: 0.05, precision: 2 }
    );
    markTransient(strengthWidget);

    const boxesWidget = node.addWidget("text", `LoRA ${idx + 1} boxes`, sel.boxes ?? `${idx + 1}`, (value) => {
      updateSelection(node, idx, (s) => { s.boxes = value; });
    });
    boxesWidget.options = boxesWidget.options || {};
    boxesWidget.options.placeholder = "e.g. 1,3-5";
    markTransient(boxesWidget);

    const colorWidget = node.addWidget("text", `LoRA ${idx + 1} color`, sel.color || DEFAULT_COLORS[idx % DEFAULT_COLORS.length], (value) => {
      updateSelection(node, idx, (s) => { s.color = value; });
    });
    colorWidget.options = colorWidget.options || {};
    colorWidget.options.placeholder = "#rrggbb";
    markTransient(colorWidget);

    const removeWidget = node.addWidget("button", `remove LoRA ${idx + 1}`, null, () => {
      const selections = readSelections(node);
      selections.splice(idx, 1);
      writeSelections(node, selections);
      rebuildRows(node);
    });
    markTransient(removeWidget);
  });

  node.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "krea2.multi_lora_loader",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== LOADER_NODE) return;
    await ensureLoras();

    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalCreated ? originalCreated.apply(this, arguments) : undefined;
      const addWidget = this.addWidget("button", "+ Add LoRA", null, () => {
        const selections = readSelections(this);
        selections.push(defaultSelection(selections.length));
        writeSelections(this, selections);
        rebuildRows(this);
      });
      addWidget.__k2_loader_add_button = true;
      if (readSelections(this).length === 0) {
        writeSelections(this, [defaultSelection(0), defaultSelection(1)]);
      }
      setTimeout(() => rebuildRows(this), 0);
      return result;
    };

    const originalConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = originalConfigure ? originalConfigure.apply(this, arguments) : undefined;
      setTimeout(() => rebuildRows(this), 0);
      return result;
    };
  },
});
