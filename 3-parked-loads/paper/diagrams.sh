#!/bin/bash
# PowerPoint diagram -> PDF (LibreOffice) -> cropped to its drawing (Ghostscript), for the paper.
# Usage: bash paper/diagrams.sh   (NODE_PATH must point to a node_modules with pptxgenjs)
set -e
cd "$(dirname "$0")"
node fig_idea.js
tmp=$(mktemp -d)
soffice --headless --convert-to pdf --outdir "$tmp" fig_idea.pptx >/dev/null
bb=$(gs -q -dBATCH -dNOPAUSE -sDEVICE=bbox "$tmp/fig_idea.pdf" 2>&1 | grep HiResBoundingBox | awk '{print $2, $3, $4, $5}')
read x0 y0 x1 y1 <<< "$bb"
m=1
w=$(echo "$x1 - $x0 + 2*$m" | bc); h=$(echo "$y1 - $y0 + 2*$m" | bc)
gs -q -dBATCH -dNOPAUSE -sDEVICE=pdfwrite -dEmbedAllFonts=true -dSubsetFonts=true \
   -dDEVICEWIDTHPOINTS="$w" -dDEVICEHEIGHTPOINTS="$h" -dFIXEDMEDIA \
   -o fig_idea.pdf -c "<</PageOffset [$(echo "$m - $x0" | bc) $(echo "$m - $y0" | bc)]>> setpagedevice" \
   -f "$tmp/fig_idea.pdf"
rm -rf "$tmp"
echo "fig_idea.pdf: ${w} x ${h} pt"
