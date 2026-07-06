import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_TYPE = "Krea2RegionalLoRAMasks";
const JSON_WIDGET = "regions_json";
let LORA_LIST = ["None"];

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
    console.warn("[Krea2RegionalLoRAMasks] Could not fetch LoRA list", err);
  }
  return LORA_LIST;
}

function jsonWidget(node) {
  return node.widgets?.find((w) => w.name === JSON_WIDGET);
}

function readRegions(node) {
  const w = jsonWidget(node);
  if (!w) return [];
  try {
    const parsed = JSON.parse(w.value || "[]");
    return Array.isArray(parsed) ? parsed : [];
  } catch (_) {
    return [];
  }
}

function writeRegions(node, regions) {
  const w = jsonWidget(node);
  if (!w) return;
  w.value = JSON.stringify(regions, null, 2);
  if (w.inputEl) w.inputEl.value = w.value;
  node.setDirtyCanvas(true, true);
}

function markTransient(widget) {
  widget.__k2_region_widget = true;
  widget.serialize = false;
  widget.options = widget.options || {};
  widget.options.serialize = false;
  return widget;
}

function defaultRegion(index) {
  return {
    name: `region_${index + 1}`,
    lora: "None",
    strength: 1.0,
    enabled: true,
    bbox: { x: index === 0 ? 0.05 : 0.55, y: 0.05, w: 0.4, h: 0.85 },
  };
}

function rebuildRows(node) {
  if (!node.widgets) return;
  node.widgets = node.widgets.filter((w) => !w.__k2_region_widget);
  const regions = readRegions(node);

  regions.forEach((region, idx) => {
    const enableWidget = node.addWidget(
      "toggle",
      `region ${idx + 1} enabled`,
      region.enabled !== false && region.enable !== false,
      (value) => {
        const r = readRegions(node);
        if (r[idx]) {
          r[idx].enabled = value;
          writeRegions(node, r);
        }
      },
      { on: "on", off: "off" }
    );
    markTransient(enableWidget);

    const nameWidget = node.addWidget("text", `region ${idx + 1} name`, region.name || `region_${idx + 1}`, (value) => {
      const r = readRegions(node);
      if (r[idx]) {
        r[idx].name = value;
        writeRegions(node, r);
      }
    });
    markTransient(nameWidget);

    const loraWidget = node.addWidget(
      "combo",
      `region ${idx + 1} lora`,
      region.lora || "None",
      (value) => {
        const r = readRegions(node);
        if (r[idx]) {
          r[idx].lora = value;
          writeRegions(node, r);
        }
      },
      { values: LORA_LIST }
    );
    markTransient(loraWidget);

    const strengthWidget = node.addWidget(
      "number",
      `region ${idx + 1} strength`,
      Number.isFinite(region.strength) ? region.strength : 1.0,
      (value) => {
        const r = readRegions(node);
        if (r[idx]) {
          r[idx].strength = Number(value);
          writeRegions(node, r);
        }
      },
      { min: -5, max: 5, step: 0.05, precision: 2 }
    );
    markTransient(strengthWidget);

    const removeWidget = node.addWidget("button", `remove region ${idx + 1}`, null, () => {
      const r = readRegions(node);
      r.splice(idx, 1);
      writeRegions(node, r);
      rebuildRows(node);
    });
    markTransient(removeWidget);
  });
  node.setDirtyCanvas(true, true);
}

function getBboxCountFromSource(node) {
  const input = node.inputs?.find((i) => i.name === "bboxes");
  if (!input?.link) return null;
  const link = node.graph?.links?.[input.link];
  if (!link) return null;
  const sourceNode = node.graph?.getNodeById(link.origin_id);
  if (!sourceNode) return null;

  for (const w of sourceNode.widgets || []) {
    if (typeof w.value !== "string") continue;
    try {
      const parsed = JSON.parse(w.value);
      const arr = Array.isArray(parsed) ? parsed : parsed?.boxes || parsed?.bboxes || parsed?.regions;
      if (!Array.isArray(arr)) continue;
      if (arr.length === 0) return 0;
      const first = arr[0];
      if (first && typeof first === "object" && ("x" in first || "x0" in first || "width" in first || "w" in first || "bbox" in first)) {
        return arr.length;
      }
    } catch (_) {}
  }
  return null;
}

function syncRegionCount(node, count) {
  if (count == null) return;
  const regions = readRegions(node);
  if (regions.length === count) return;
  while (regions.length < count) regions.push(defaultRegion(regions.length));
  if (regions.length > count) regions.splice(count);
  writeRegions(node, regions);
  rebuildRows(node);
}

app.registerExtension({
  name: "krea2.regional_lora_masks_patched",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return;
    await ensureLoras();

    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalCreated ? originalCreated.apply(this, arguments) : undefined;
      this.__k2_last_bbox_count = null;
      const addWidget = this.addWidget("button", "+ Add Region", null, () => {
        const r = readRegions(this);
        r.push(defaultRegion(r.length));
        writeRegions(this, r);
        rebuildRows(this);
      });
      addWidget.__k2_add_button = true;
      setTimeout(() => rebuildRows(this), 0);
      return result;
    };

    const originalConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = originalConfigure ? originalConfigure.apply(this, arguments) : undefined;
      this.__k2_last_bbox_count = null;
      setTimeout(() => rebuildRows(this), 0);
      return result;
    };

    const originalConnections = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, index, connected, linkInfo) {
      const result = originalConnections ? originalConnections.apply(this, arguments) : undefined;
      const bboxIndex = this.inputs?.findIndex((i) => i.name === "bboxes");
      if (index === bboxIndex) {
        this.__k2_last_bbox_count = null;
        setTimeout(() => {
          const count = getBboxCountFromSource(this);
          if (count !== null) {
            this.__k2_last_bbox_count = count;
            syncRegionCount(this, count);
          }
        }, 50);
      }
      return result;
    };

    const originalDraw = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      if (originalDraw) originalDraw.apply(this, arguments);
      const count = getBboxCountFromSource(this);
      if (count !== null && count !== this.__k2_last_bbox_count) {
        this.__k2_last_bbox_count = count;
        syncRegionCount(this, count);
      }
    };
  },
});
