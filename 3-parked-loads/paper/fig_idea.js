// Fig. 1 (polling against a parked load) as a PowerPoint drawing: paper/fig_idea.pptx.
// NODE_PATH=<dir with node_modules containing pptxgenjs> node paper/fig_idea.js
// The PDF used by the letter is converted and cropped by paper/diagrams.sh.
const pptxgen = require("pptxgenjs");
const path = require("path");

const pres = new pptxgen();
pres.defineLayout({ name: "FIG", width: 3.5, height: 1.9 });
pres.layout = "FIG";
const s = pres.addSlide();
const FONT = "Times New Roman";
const K = 0.405;                                  // inches per unit of the former TikZ drawing (cm)
const X = x => (x + 1.05) * K, Y = y => (0.95 - y) * K;
const D = 1.3, W = 2.6;
const RED = "B30000", BLUE = "0000B3", GRAY = "808080", LIFE = "999999";

function text(t, x, y, opts = {}) {             // centred at drawing point (x, y)
  const w = opts.w || 1.2, h = opts.h || 0.14;
  const o = Object.assign({ fontFace: FONT, fontSize: 7, color: "000000", margin: 0, align: "center",
    valign: "middle", isTextBox: true }, opts);
  let cx = X(x), cy = Y(y);
  if (o.anchor === "east") cx -= w / 2;
  if (o.anchor === "west") cx += w / 2;
  delete o.anchor; delete o.w; delete o.h;
  s.addText(t, Object.assign({ x: cx - w / 2, y: cy - h / 2, w, h }, o));
}

function line(x1, y1, x2, y2, opts = {}) {
  const a = { x: X(x1), y: Y(y1) }, b = { x: X(x2), y: Y(y2) };
  s.addShape(pres.shapes.LINE, { x: Math.min(a.x, b.x), y: Math.min(a.y, b.y),
    w: Math.abs(b.x - a.x), h: Math.abs(b.y - a.y), flipH: b.x < a.x, flipV: b.y < a.y,
    line: Object.assign({ color: "000000", width: 0.6 }, opts) });
}
const arrow = (x1, y1, x2, y2, color = "000000", width = 0.6) =>
  line(x1, y1, x2, y2, { color, width, endArrowType: "triangle" });

function sloped(t, x1, y1, x2, y2, side, color) {  // label along an arrow, above or below it
  const dx = X(x2) - X(x1), dy = Y(y2) - Y(y1);
  let ang = Math.atan2(dy, dx) * 180 / Math.PI;
  if (ang > 90) ang -= 180; if (ang < -90) ang += 180;
  const n = side === "above" ? -0.075 : 0.075;
  const r = ang * Math.PI / 180;
  const cx = (X(x1) + X(x2)) / 2 - n * Math.sin(r), cy = (Y(y1) + Y(y2)) / 2 + n * Math.cos(r);
  s.addText(t, { x: cx - 0.3, y: cy - 0.07, w: 0.6, h: 0.14, rotate: (ang + 360) % 360, fontFace: FONT,
    fontSize: 7, color, margin: 0, align: "center", valign: "middle", isTextBox: true });
}

function brace(y1, y2, ox) {                    // left brace at x = -0.1 between y1 (top) and y2
  s.addShape(pres.shapes.LEFT_BRACE, { x: X(ox - 0.1) - 0.045, y: Y(y1), w: 0.045, h: Y(y2) - Y(y1),
    line: { color: "000000", width: 0.6 } });
  text("hand-\noff", ox - 0.42, (y1 + y2) / 2, { w: 0.5, h: 0.24, align: "right", anchor: "east" });
}

for (const [ox, ttl] of [[0, "(a) Polling"], [4.3, "(b) Parked load"]]) {
  text(ttl, ox + D, 0.75, { fontSize: 8, bold: true });
  for (const [x, n] of [[0, "waiter"], [D, "device"], [W, "writer"]]) {
    text(n, ox + x, 0.35, { w: 0.6 });
    line(ox + x, 0.2, ox + x, -2.62, { color: LIFE, width: 0.5, dashType: "dash" });
  }
}

// (a) polling
for (const y of [-0.2, -0.85, -1.5]) {
  arrow(0, y, D, y - 0.22);
  arrow(D, y - 0.22, 0, y - 0.44, GRAY);
}
text("poll", 0.35, -0.1, { color: GRAY, w: 0.4 });
text("old", -0.05, -0.64, { color: GRAY, w: 0.3, anchor: "east", align: "right" });
arrow(W, -1.45, D, -1.8, RED, 1.2);
sloped("write", W, -1.45, D, -1.8, "above", RED);
arrow(0, -2.15, D, -2.37);
arrow(D, -2.37, 0, -2.59, BLUE, 1.2);
sloped("new", D, -2.37, 0, -2.59, "below", BLUE);
brace(-1.8, -2.59, 0);
text("one device read per poll", D, -2.95, { w: 1.6 });

// (b) parked load
const o = 4.3;
arrow(o, -0.2, o + D, -0.42);
s.addShape(pres.shapes.RECTANGLE, { x: X(o + D - 0.08), y: Y(-0.42), w: 0.16 * K, h: (1.8 - 0.42) * K,
  fill: { color: "FFD2A6" }, line: { color: "FFD2A6", width: 0 } });
text("parked;\ncredit\nfreed", o + D + 0.12, -0.8, { w: 0.5, h: 0.36, align: "left", anchor: "west" });
arrow(o + W, -1.45, o + D, -1.8, RED, 1.2);
sloped("write", o + W, -1.45, o + D, -1.8, "above", RED);
arrow(o + D, -1.8, o, -2.02, BLUE, 1.2);
sloped("new", o + D, -1.8, o, -2.02, "below", BLUE);
brace(-1.8, -2.02, o);
text("answered within τ = 25 µs,\nelse “not yet”", o + D, -2.8, { w: 1.6, h: 0.26 });

pres.writeFile({ fileName: path.join(__dirname, "fig_idea.pptx") }).then(f => console.log("wrote", f));
