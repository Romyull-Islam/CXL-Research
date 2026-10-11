"""Use the paper's body font in matplotlib figures.

matplotlib embeds OpenType (CFF) fonts with a font-type mismatch that PDF checkers flag, so the
OTF files are converted once to TrueType outlines in ~/.cache/figfonts, renamed to "<family> TT"
and registered. Figures then embed clean Type 42 fonts.
"""
from pathlib import Path

from matplotlib import font_manager, rcParams

CACHE = Path.home() / ".cache" / "figfonts"


def _to_ttf(src, dst, family):
    from fontTools.pens.cu2quPen import Cu2QuPen
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    from fontTools.ttLib import TTFont, newTable
    f = TTFont(src)
    order = f.getGlyphOrder()
    gs = f.getGlyphSet()
    glyphs = {}
    for name in order:
        pen = TTGlyphPen(gs)
        gs[name].draw(Cu2QuPen(pen, 1.0, reverse_direction=True))
        glyphs[name] = pen.glyph()
    f["loca"] = newTable("loca")
    f["glyf"] = glyf = newTable("glyf")
    glyf.glyphOrder, glyf.glyphs = order, glyphs
    del f["CFF "]
    if "VORG" in f:
        del f["VORG"]
    glyf.compile(f)
    for name, g in glyf.glyphs.items():
        if hasattr(g, "xMin"):
            f["hmtx"][name] = (f["hmtx"][name][0], g.xMin)
    f["maxp"] = maxp = newTable("maxp")
    maxp.tableVersion = 0x00010000
    for k in ("maxZones", "maxTwilightPoints", "maxStorage", "maxFunctionDefs", "maxInstructionDefs",
              "maxStackElements", "maxSizeOfInstructions", "maxComponentElements"):
        setattr(maxp, k, 1 if k == "maxZones" else 0)
    maxp.compile(f)
    post = f["post"]
    post.formatType, post.extraNames, post.mapping, post.glyphOrder = 2.0, [], {}, order
    style = f["name"].getDebugName(2) or "Regular"
    for rec in f["name"].names:
        if rec.nameID in (1, 16):
            rec.string = family
        elif rec.nameID == 4:
            rec.string = f"{family} {style}"
        elif rec.nameID == 6:
            rec.string = f"{family}-{style}".replace(" ", "")
    f.sfntVersion = "\000\001\000\000"
    f.save(dst)


def use(files, family):
    """files: OTF paths of one family (regular, italic, bold); family: the name to register."""
    CACHE.mkdir(parents=True, exist_ok=True)
    for src in files:
        dst = CACHE / (Path(src).stem + ".ttf")
        if not dst.exists():
            _to_ttf(src, dst, family)
        font_manager.fontManager.addfont(str(dst))
    rcParams.update({"font.family": family, "mathtext.fontset": "custom", "mathtext.rm": family,
                     "mathtext.it": f"{family}:italic", "mathtext.bf": f"{family}:bold",
                     "pdf.fonttype": 42})


URW = "/usr/share/fonts/opentype/urw-base35/"
LIB = "/usr/share/fonts/opentype/linux-libertine/"
TIMES = [URW + f"NimbusRoman-{s}.otf" for s in ("Regular", "Italic", "Bold")]
LIBERTINE = [LIB + f"LinLibertine_{s}.otf" for s in ("R", "RI", "RB")]
