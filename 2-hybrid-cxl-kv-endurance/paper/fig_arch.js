// System figure: GPU node, host, hybrid CXL device and the discard path, as a PowerPoint drawing.
// NODE_PATH=<dir with node_modules containing pptxgenjs> node paper/fig_arch.js; PDF via paper/diagrams.sh
const pptxgen = require("pptxgenjs");
const path = require("path");

const pres = new pptxgen();
pres.defineLayout({ name: "FIG", width: 5.5, height: 2.1 });
pres.layout = "FIG";
const s = pres.addSlide();
const FONT = "Times New Roman", LINE = "404040";

function box(t, x, y, w, h, fill, opts = {}) {
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: LINE, width: 0.6 } });
  s.addText(t, Object.assign({ x, y, w, h, fontFace: FONT, fontSize: 7, color: "000000", margin: 0.02,
    align: "center", valign: "middle", isTextBox: true }, opts));
}
function text(t, x, y, w, h, opts = {}) {
  s.addText(t, Object.assign({ x, y, w, h, fontFace: FONT, fontSize: 6.5, color: "000000", margin: 0,
    align: "center", valign: "middle", isTextBox: true }, opts));
}
function arrow(x1, y1, x2, y2, opts = {}) {
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1),
    h: Math.abs(y2 - y1), flipH: x2 < x1, flipV: y2 < y1,
    line: Object.assign({ color: LINE, width: 0.75, endArrowType: "triangle" }, opts) });
}
function seg(x1, y1, x2, y2, opts = {}) {
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1),
    h: Math.abs(y2 - y1), flipH: x2 < x1, flipV: y2 < y1, line: Object.assign({ color: LINE, width: 0.75 }, opts) });
}

// node and host
box("8-GPU prefill node\ncomputes KV;\nHBM KV cache", 0.05, 0.55, 0.85, 0.75, "E8EEF7");
box("host:\nserving engine,\nKV block allocator", 1.3, 0.55, 0.95, 0.75, "EEEEEE");
arrow(0.9, 0.98, 1.3, 0.98);
text("evict\nKV block", 0.9, 0.68, 0.4, 0.26, { fontSize: 6 });

// device
s.addShape(pres.shapes.RECTANGLE, { x: 2.55, y: 0.05, w: 2.9, h: 1.6, fill: { color: "FFFFFF" },
  line: { color: "7F7F7F", width: 0.6, dashType: "solid" } });
text("hybrid CXL Type-3 device", 2.55, 0.08, 2.9, 0.16, { fontSize: 7, bold: true });
box("DRAM cache\n48 GB, LRU", 2.7, 0.38, 1.05, 0.5, "DCE9F5");
box("NAND\n2 TB TLC or\n667 GB pseudo-SLC", 4.25, 0.38, 1.1, 0.5, "FBE5D6");
box("admission\nall / 2nd hit / oracle", 2.7, 1.08, 1.05, 0.45, "FFF2CC");
box("FTL: page map,\ngreedy GC", 4.25, 1.08, 1.1, 0.45, "E2F0D9");

text("CXL.mem\nloads, stores", 2.0, 0.3, 0.5, 0.25, { fontSize: 6 });
arrow(2.25, 0.62, 2.7, 0.62);
arrow(3.22, 0.88, 3.22, 1.08);                                // DRAM evicts -> admission
text("evict", 3.27, 0.9, 0.3, 0.16, { fontSize: 6, align: "left" });
arrow(3.75, 1.3, 4.25, 1.3);                                  // admission -> FTL
text("write", 3.75, 1.14, 0.5, 0.15, { fontSize: 6 });
arrow(4.8, 1.08, 4.8, 0.88);                                  // FTL -> NAND
text("program", 4.83, 0.9, 0.45, 0.16, { fontSize: 6, align: "left" });
arrow(4.25, 0.52, 3.75, 0.52);                                // NAND hit -> DRAM
text("hit: promote", 3.75, 0.35, 0.5, 0.15, { fontSize: 6 });

// discard path on the management interface
const D = { color: "C00000", width: 0.9, dashType: "dash" };
seg(1.78, 1.3, 1.78, 1.88, D);
seg(1.78, 1.88, 4.8, 1.88, D);
arrow(4.8, 1.88, 4.8, 1.53, D);
seg(2.25, 1.12, 2.5, 1.12, D);                                // host -> DRAM: drop dead KV
arrow(2.5, 1.12, 2.7, 0.8, D);
text("per-request discard (CXL.io mailbox): DRAM drops dead KV, FTL trims it, device acknowledges",
  1.85, 1.92, 3.6, 0.14, { fontSize: 6.5, color: "C00000", align: "left" });

pres.writeFile({ fileName: path.join(__dirname, "fig_arch.pptx") }).then(f => console.log("wrote", f));
