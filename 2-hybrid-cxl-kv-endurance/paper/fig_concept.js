// Fig. 1 (how KV dies) as a PowerPoint drawing: paper/fig_concept.pptx.
// node paper/fig_concept.js  (needs pptxgenjs: NODE_PATH=<dir with node_modules> node ...)
// The PDF used by the paper is converted and cropped by paper/diagrams.sh.
const pptxgen = require("pptxgenjs");
const path = require("path");

const pres = new pptxgen();
pres.defineLayout({ name: "FIG", width: 5.5, height: 1.65 });
pres.layout = "FIG";
const s = pres.addSlide();
const FONT = "Times New Roman";
const GRID = "666666";
const COL = { A: "4C72B0", B: "DD8452", C: "55A868" };
// write order shades for panel (a), EB2 then EB3 (matplotlib Blues as before)
const BLUES = ["5DA5D1", "549FCD", "4A98C9", "4090C5", "3989C1", "3181BD", "2979B9", "2070B4",
               "1B69AF", "1561A9", "0E59A2", "09529D", "084A91", "084A91", "084A91", "084A91"];
const TITLE = ["(a) LRU prefix cache", "(b) per request, TRIM", "(c) per request, no TRIM"];
const NOTE = ["evicted in write order: EB1 empties", "request B ends: holes in every EB",
              "B’s pages dead but still valid"];
const WAF = ["measured WAF 1.00 to 1.02", "measured WAF 2.1", "measured WAF 5.2"];

const PW = 5.5 / 3, PG = 0.165, GAP = 0.022, PH = 0.27, RG = 0.06, Y0 = 0.40;

function text(t, x, y, w, h, size, opts = {}) {
  s.addText(t, Object.assign({ x, y, w, h, fontFace: FONT, fontSize: size, color: "000000", margin: 0,
    align: "center", valign: "middle", isTextBox: true }, opts));
}

function hatch(x0, y0, w, h) {
  const sp = 0.045;
  for (let c = -h; c < w; c += sp) {
    const ua = Math.max(0, c), ub = Math.min(w, c + h);
    if (ub - ua < 0.004) continue;
    const xa = x0 + ua, ya = y0 + h - (ua - c), xb = x0 + ub, yb = y0 + h - (ub - c);
    s.addShape(pres.shapes.LINE, { x: xa, y: yb, w: xb - xa, h: ya - yb, flipV: true,
      line: { color: GRID, width: 0.5 } });
  }
}

for (let k = 0; k < 3; k++) {
  const px = k * PW, x0 = px + 0.25, xc = x0 + (8 * PG + 7 * GAP) / 2;
  text(TITLE[k], xc - 0.9, 0.02, 1.8, 0.16, 7.5);
  text(NOTE[k], xc - 0.9, 0.20, 1.8, 0.14, 6);
  for (let e = 0; e < 3; e++) {
    const y = Y0 + e * (PH + RG);
    text(`EB${e + 1}`, px + 0.0, y, 0.22, PH, 6.5, { align: "right" });
    for (let j = 0; j < 8; j++) {
      const x = x0 + j * (PG + GAP);
      let face, dead;
      if (k === 0) { dead = e === 0; face = dead ? "FFFFFF" : BLUES[(e - 1) * 8 + j]; }
      else { const o = "ABC"[(j + e) % 3]; dead = o === "B"; face = COL[o]; }
      const trimmed = dead && k < 2;
      s.addShape(pres.shapes.RECTANGLE, { x, y, w: PG, h: PH, fill: { color: trimmed ? "FFFFFF" : face },
        line: { color: GRID, width: 0.5 } });
      if (trimmed) hatch(x, y, PG, PH);
      if (dead && k === 2) text("x", x, y, PG, PH, 7, { color: "FFFFFF" });
    }
  }
  text(WAF[k], xc - 0.9, Y0 + 3 * PH + 2 * RG + 0.08, 1.8, 0.16, 7);
}

pres.writeFile({ fileName: path.join(__dirname, "fig_concept.pptx") }).then(f => console.log("wrote", f));
