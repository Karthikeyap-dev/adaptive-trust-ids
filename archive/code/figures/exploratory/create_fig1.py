from pathlib import Path
import cairosvg

# ============================================================
# ESWA FIGURE 1 — SCIENTIFIC VECTOR ARCHITECTURE
# ============================================================

W, H = 1800, 1200

svg_path = Path("fig1_ESWA_scientific_vector.svg")
pdf_path = Path("fig1_ESWA_scientific_vector.pdf")
png_path = Path("fig1_ESWA_scientific_4k.png")


def esc(text):
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )


SVG = [
f'''<svg xmlns="http://www.w3.org/2000/svg"
      width="{W}"
      height="{H}"
      viewBox="0 0 {W} {H}">

<defs>

    <!-- BLACK ARROW -->
    <marker id="blackArrow"
            markerWidth="14"
            markerHeight="14"
            refX="11"
            refY="7"
            orient="auto">
        <path d="M0,0 L14,7 L0,14 Z"
              fill="#111111"/>
    </marker>

    <!-- RED ARROW -->
    <marker id="redArrow"
            markerWidth="14"
            markerHeight="14"
            refX="11"
            refY="7"
            orient="auto">
        <path d="M0,0 L14,7 L0,14 Z"
              fill="#d92828"/>
    </marker>

</defs>

<!-- WHITE BACKGROUND -->
<rect width="1800"
      height="1200"
      fill="#ffffff"/>

'''
]


# ============================================================
# DRAWING HELPERS
# ============================================================

def box(x, y, w, h, fill, stroke, radius=24):

    SVG.append(
        f'<rect x="{x}" y="{y}" '
        f'width="{w}" height="{h}" '
        f'rx="{radius}" '
        f'fill="{fill}" '
        f'stroke="{stroke}" '
        f'stroke-width="4"/>'
    )


def text(x, y, value, size=28, weight=600):

    SVG.append(
        f'<text x="{x}" y="{y}" '
        f'text-anchor="middle" '
        f'font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}px" '
        f'font-weight="{weight}" '
        f'fill="#111111">'
        f'{esc(value)}'
        f'</text>'
    )


def line(
    x1, y1,
    x2, y2,
    color="#111111",
    width=5,
    dash=None,
    marker=None
):

    dash_attr = (
        f' stroke-dasharray="{dash}"'
        if dash else ""
    )

    marker_attr = (
        f' marker-end="url(#{marker})"'
        if marker else ""
    )

    SVG.append(
        f'<line '
        f'x1="{x1}" y1="{y1}" '
        f'x2="{x2}" y2="{y2}" '
        f'stroke="{color}" '
        f'stroke-width="{width}" '
        f'stroke-linecap="round"'
        f'{dash_attr}'
        f'{marker_attr}/>'
    )


def polyline(
    points,
    color="#111111",
    width=5,
    dash=None,
    marker=None
):

    point_string = " ".join(
        f"{x},{y}" for x, y in points
    )

    dash_attr = (
        f' stroke-dasharray="{dash}"'
        if dash else ""
    )

    marker_attr = (
        f' marker-end="url(#{marker})"'
        if marker else ""
    )

    SVG.append(
        f'<polyline '
        f'points="{point_string}" '
        f'fill="none" '
        f'stroke="{color}" '
        f'stroke-width="{width}" '
        f'stroke-linecap="round" '
        f'stroke-linejoin="round"'
        f'{dash_attr}'
        f'{marker_attr}/>'
    )


# ============================================================
# MAIN COLUMN GEOMETRY
# ============================================================

CENTER_X = 720

MAIN_X = 250
MAIN_WIDTH = 940
MAIN_HEIGHT = 165

Y1 = 55
Y2 = 285
Y3 = 515
Y4 = 745


# ============================================================
# MAIN FOUR BOXES
# ============================================================

# 1. Network alert
box(
    MAIN_X, Y1,
    MAIN_WIDTH, MAIN_HEIGHT,
    "#f5f5f5",
    "#666666"
)

# 2. Predictive Agent
box(
    MAIN_X, Y2,
    MAIN_WIDTH, MAIN_HEIGHT,
    "#eaf2ff",
    "#245bd1"
)

# 3. Adaptive Trust Engine
box(
    MAIN_X, Y3,
    MAIN_WIDTH, MAIN_HEIGHT,
    "#f0e8ff",
    "#6836a3"
)

# 4. Constrained Policy Optimization
box(
    MAIN_X, Y4,
    MAIN_WIDTH, MAIN_HEIGHT,
    "#f0e8ff",
    "#6836a3"
)


# ============================================================
# BOTTOM DECISION BOXES
# ============================================================

# Auto execute
box(
    105, 975,
    500, 145,
    "#effaf1",
    "#269b46",
    20
)

# Human review
box(
    650, 975,
    620, 145,
    "#fff0f0",
    "#df3030",
    20
)

# Human feedback
box(
    1370, 965,
    360, 155,
    "#f5f5f5",
    "#666666",
    20
)


# ============================================================
# ICON 1 — SERVER
# ============================================================

SVG.append(
'''<g transform="translate(315,95)">
    <rect width="72"
          height="90"
          rx="6"
          fill="#333333"/>
'''
)

for yy in (19, 45, 71):

    SVG.append(
        f'<line x1="12" y1="{yy}" '
        f'x2="52" y2="{yy}" '
        f'stroke="white" '
        f'stroke-width="5" '
        f'stroke-linecap="round"/>'
    )

    SVG.append(
        f'<circle cx="61" cy="{yy}" '
        f'r="3" fill="white"/>'
    )

SVG.append("</g>")


# ============================================================
# ICON 2 — CPU / CHIP
# ============================================================

SVG.append(
'''<g transform="translate(315,325)"
   fill="none"
   stroke="#245bd1"
   stroke-linecap="round">

<rect x="10"
      y="10"
      width="72"
      height="72"
      rx="10"
      stroke-width="7"/>

<rect x="31"
      y="31"
      width="30"
      height="30"
      rx="4"
      fill="#245bd1"
      stroke="none"/>
'''
)

for p in (21, 36, 51, 66):

    SVG.append(
        f'<line x1="{p}" y1="10" '
        f'x2="{p}" y2="-7" '
        f'stroke-width="5"/>'
    )

    SVG.append(
        f'<line x1="{p}" y1="82" '
        f'x2="{p}" y2="99" '
        f'stroke-width="5"/>'
    )

    SVG.append(
        f'<line x1="10" y1="{p}" '
        f'x2="-7" y2="{p}" '
        f'stroke-width="5"/>'
    )

    SVG.append(
        f'<line x1="82" y1="{p}" '
        f'x2="99" y2="{p}" '
        f'stroke-width="5"/>'
    )

SVG.append("</g>")


# ============================================================
# ICON 3 — SHIELD
# ============================================================

SVG.append(
'''<g transform="translate(315,555)"
   fill="none"
   stroke="#6836a3"
   stroke-width="8"
   stroke-linejoin="round">

<path d="
M45 5
L87 25
L80 80
C74 109 45 123 45 123
C45 123 16 109 10 80
L3 25 Z"/>

<path d="
M22 63
L39 80
L69 45"
stroke-linecap="round"/>

</g>'''
)


# ============================================================
# ICON 4 — GAUGE
# ============================================================

SVG.append(
'''<g transform="translate(315,780)"
   fill="none"
   stroke="#6836a3"
   stroke-linecap="round">

<path d="
M0 82
A62 62 0 0 1 124 82"
stroke-width="8"/>

<line x1="9" y1="67"
      x2="20" y2="64"
      stroke-width="6"/>

<line x1="28" y1="32"
      x2="35" y2="42"
      stroke-width="6"/>

<line x1="62" y1="21"
      x2="62" y2="34"
      stroke-width="6"/>

<line x1="96" y1="32"
      x2="89" y2="42"
      stroke-width="6"/>

<line x1="115" y1="67"
      x2="104" y2="64"
      stroke-width="6"/>

<line x1="62" y1="82"
      x2="97" y2="43"
      stroke-width="7"/>

<circle cx="62"
        cy="82"
        r="7"
        fill="#6836a3"
        stroke="none"/>

</g>'''
)


# ============================================================
# ICON 5 — GREEN CHECK
# ============================================================

SVG.append(
'''<g transform="translate(170,1048)">

<circle r="36"
        fill="#269b46"/>

<path d="
M-18 0
L-5 15
L21 -18"
fill="none"
stroke="white"
stroke-width="9"
stroke-linecap="round"
stroke-linejoin="round"/>

</g>'''
)


# ============================================================
# ICON 6 — HUMAN
# ============================================================

SVG.append(
'''<g transform="translate(725,1035)"
   fill="#df3030">

<circle cx="30"
        cy="16"
        r="19"/>

<path d="
M-3 80
C0 47 13 34 30 34
C47 34 60 47 63 80 Z"/>

</g>'''
)


# ============================================================
# ICON 7 — FEEDBACK
# ============================================================

SVG.append(
'''<g transform="translate(1415,1008)"
   fill="none"
   stroke="#333333"
   stroke-width="8"
   stroke-linecap="round">

<path d="
M0 48
A48 48 0 0 1 72 8"/>

<path d="
M72 8
L70 27
L53 18"/>

<path d="
M72 48
A48 48 0 0 1 0 88"/>

<path d="
M0 88
L2 69
L19 78"/>

</g>'''
)


# ============================================================
# TEXT
# ============================================================

# Network alert
text(
    800, 120,
    "Network alert",
    29, 700
)

text(
    800, 157,
    "(benchmark dataset)",
    25, 600
)


# Predictive Agent
text(
    800, 350,
    "Predictive Agent",
    29, 700
)

text(
    800, 387,
    "(ML classifier)",
    25, 600
)


# Adaptive Trust Engine
text(
    800, 565,
    "Adaptive Trust Engine —",
    27, 700
)

text(
    800, 602,
    "Beta(a,b) per (agent, category),",
    25, 600
)

text(
    800, 638,
    "decay-weighted",
    25, 600
)


# Policy Optimization
text(
    800, 795,
    "Constrained Policy Optimization —",
    27, 700
)

text(
    800, 832,
    "Arbitration threshold tuning",
    25, 600
)


# Auto execute
text(
    390, 1062,
    "Auto-execute",
    28, 700
)


# Human analyst
text(
    1010, 1037,
    "Human analyst review",
    26, 700
)

text(
    1010, 1075,
    "(SHAP + LLM explanation)",
    24, 600
)


# Feedback
text(
    1560, 1025,
    "Human feedback",
    24, 700
)

text(
    1560, 1062,
    "(correct / incorrect)",
    23, 600
)


# ============================================================
# MAIN FLOW ARROWS
# ============================================================

# Alert -> Predictive Agent
line(
    CENTER_X, 220,
    CENTER_X, 285,
    marker="blackArrow"
)

# Predictive Agent -> Trust
line(
    CENTER_X, 450,
    CENTER_X, 515,
    marker="blackArrow"
)

# Trust -> Optimization
line(
    CENTER_X, 680,
    CENTER_X, 745,
    marker="blackArrow"
)


# ============================================================
# DECISION SPLIT
# ============================================================

# Vertical line leaving optimizer
line(
    CENTER_X, 910,
    CENTER_X, 945
)

# Left branch
polyline(
    [
        (CENTER_X, 945),
        (355, 945),
        (355, 975)
    ],
    marker="blackArrow"
)

# Right branch
polyline(
    [
        (CENTER_X, 945),
        (960, 945),
        (960, 975)
    ],
    marker="blackArrow"
)


# ============================================================
# FEEDBACK LOOP
# ============================================================

# Human review -> Human feedback
line(
    1270, 1048,
    1370, 1048,
    color="#d92828",
    width=5,
    dash="12 10",
    marker="redArrow"
)


# Human feedback -> Adaptive Trust Engine
#
# Routed completely outside the main pipeline.
#
polyline(
    [
        (1550, 965),
        (1550, 455),
        (1230, 455),
        (1230, 598),
        (1190, 598)
    ],
    color="#d92828",
    width=5,
    dash="12 10",
    marker="redArrow"
)


# ============================================================
# CLOSE SVG
# ============================================================

SVG.append("</svg>")


svg_content = "\n".join(SVG)

svg_path.write_text(
    svg_content,
    encoding="utf-8"
)


# ============================================================
# EXPORT
# ============================================================

# True vector PDF
cairosvg.svg2pdf(
    bytestring=svg_content.encode("utf-8"),
    write_to=str(pdf_path)
)

# 4K PNG preview
cairosvg.svg2png(
    bytestring=svg_content.encode("utf-8"),
    write_to=str(png_path),
    output_width=3840,
    output_height=2560
)


print()
print("==============================================")
print("ESWA FIGURE 1 CREATED")
print("==============================================")
print()
print("Vector SVG:")
print(svg_path)
print()
print("Vector PDF:")
print(pdf_path)
print()
print("4K PNG:")
print(png_path)
print()
print("The PDF/SVG are true vector artwork.")
print("The PNG is only a high-resolution preview.")