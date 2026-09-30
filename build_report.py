"""Assemble the PDF report from charts/*.png. Edit AUTHOR/CONTACT below, then: python3 build_report.py"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

AUTHOR = "Kelly Todhunter"
CONTACT = "kellytodhunter@gmail.com  |  linkedin.com/in/kelly-todhunter-906b8830b/  |  github.com/kellytodhunter/ducks-defense"

ROOT = Path(__file__).parent
CH = ROOT / "charts"
OUT = ROOT / "Ducks_5v5_Defense_Analysis.pdf"
ORANGE, INK, SUB = colors.HexColor("#eb6834"), colors.HexColor("#0b0b0b"), colors.HexColor("#52514e")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=20, leading=24, alignment=TA_LEFT, textColor=INK, spaceAfter=4)
BY = ParagraphStyle("BY", parent=ss["Normal"], fontSize=9, textColor=SUB, spaceAfter=10)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=12.5, textColor=INK, spaceBefore=8, spaceAfter=4, keepWithNext=1)
P = ParagraphStyle("P", parent=ss["Normal"], fontSize=9.6, leading=13.6, textColor=INK, spaceAfter=5)
B = ParagraphStyle("B", parent=P, leftIndent=12, bulletIndent=0, spaceAfter=3)
CAP = ParagraphStyle("CAP", parent=P, fontSize=8.3, leading=11, textColor=SUB, spaceAfter=8)
SM = ParagraphStyle("SM", parent=P, fontSize=8.6, leading=12)


def fig(name, width, caption):
    from PIL import Image as PI
    w, h = PI.open(CH / name).size
    return KeepTogether([Image(str(CH / name), width=width, height=width * h / w), Paragraph(caption, CAP)])


def bullets(items, style=B):
    return [Paragraph(t, style, bulletText="•") for t in items]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(SUB)
    canvas.drawString(0.75 * inch, 0.5 * inch, f"{AUTHOR}  |  Data: NHL public API, 2025-26 regular season (all 1,312 games), 5v5 unless noted")
    canvas.drawRightString(letter[0] - 0.75 * inch, 0.5 * inch, f"Page {doc.page}")
    canvas.restoreState()


story = [
    Paragraph("Where the Ducks' 5v5 defense leaks", H1),
    Paragraph(f"A location and chance-source breakdown of 2025-26 &nbsp;|&nbsp; {AUTHOR} &nbsp;|&nbsp; {CONTACT}", BY),

    Paragraph("The short version", H2),
    *bullets([
        "<b>The problem is chance quality, not shot volume.</b> At 5v5 the Ducks ranked 28th in expected goals against per 60 "
        "(2.77 vs 2.60 league average) and 29th in goals against per 60. Attempts against are only 21st, and the Ducks are "
        "<b>5th best</b> at limiting shots from the point.",
        "<b>Roughly three-quarters of the excess is in the slot.</b> The Ducks allow about +0.18 xGA/60 more than an average team "
        "(about 11 goals over a season). Slot chances account for +0.14 of that (95% CI +0.02 to +0.26). "
        "The crease and the perimeter are league-average. Counting slot and crease together the excess is +0.13, but its interval (-0.01 to +0.28) "
        "just includes zero, so the evidence is consistent in direction rather than conclusive.",
        "<b>It looks more systemic than personnel, though that test is low-powered.</b> The pattern holds at home and on the road and in both halves of the season. "
        "On-ice/off-ice results for 20 skaters are statistically indistinguishable from noise "
        "(heterogeneity test p = 0.66; defensemen and forwards look alike).",
    ]),
    Spacer(1, 4),
    fig("1_rink_map.png", 2.9 * inch,
        "Figure 1. Ducks minus league-average unblocked attempts against per 60 (5v5), smoothed. The extra chances are concentrated "
        "at the net-front and in the high slot, not on the perimeter."),
    fig("2_zone_excess.png", 4.3 * inch,
        "Figure 2. Excess xG against per 60 by location (slot = inside the faceoff dots, excluding the crease), 95% intervals over games. "
        "Only the slot interval excludes zero; zone boundaries were my choice and not corrected for multiple comparisons."),

    Paragraph("Where the extra slot chances come from", H2),
    Paragraph(
        "Classifying each shot by how the chance began (from the sequence of events before it), rebound and forecheck-turnover chances against the Ducks "
        "are not distinguishable from league average. Most of the point estimate sits in <b>settled offensive-zone play</b> (roughly 70% of it), with smaller "
        "contributions from rushes and shots soon after faceoffs. Every interval here includes or touches zero, so read the direction, not the decimals.", P),
    fig("3_source.png", 5.2 * inch,
        "Figure 3. Slot + crease xG against per 60 vs league, by chance source (note: this combined region differs from Figure 2's slot-only). Definitions are heuristics from the event feed, which has no puck tracking."),

    Paragraph("Robustness", H2),
    fig("4_consistency.png", 3.8 * inch, "Figure 4. The slot + crease excess has the same sign in every split I tried."),
    fig("5_players.png", 3.7 * inch,
        "Figure 5. On-ice minus off-ice slot + crease xGA/60 for each Ducks skater. One of 20 intervals clears zero, which is about what chance "
        "alone produces. Single-season on/off is low-powered and confounded by usage and linemates."),

    Paragraph("What I would look at next", H2),
    *bullets([
        "<b>Pull every slot attempt against in settled offensive-zone play</b> and tag the breakdown: weak-side/back-door seams, "
        "defensive-zone collapse leaving the middle open, or low-forward coverage. Public event data cannot separate these; "
        "tracking or video can. Because no individual stands out, I would look for a shared coverage rule before looking for a player.",
        "<b>Rush defense through the middle lane.</b> The slot excess off the rush is small and its interval touches zero, but it is "
        "directionally the same problem and the cheapest to check on tape.",
        "<b>Defensive-zone faceoff plays.</b> Chances within 12 seconds of an opponent faceoff win skew high; worth checking against "
        "the staff's own set-play data.",
    ]),
    Paragraph(
        "<b>Goaltending context.</b> The Ducks' 5v5 goals against ran 12.5 above expected. Starter Lukas Dostal was exactly average "
        "(0.0 goals saved above expected per 100 shots over 1,838 shots); the two backups combined for about -11.7 on 909 shots. "
        "Those are small samples, but it means part of the goals-against ranking reflects goaltending rather than skater defense.", P),

    Paragraph("Method and caveats", H2),
    *bullets([
        "<b>Data:</b> NHL public play-by-play and shift-chart endpoints for all 1,312 regular-season games (league baseline) and the Ducks' 82. "
        "Rates use 5v5 game-clock time reconstructed from situation codes; it sums to 3,600 s per regulation game.",
        "<b>xG model:</b> gradient-boosted trees on location, angle, chance source, strength, score and period; out-of-fold predictions "
        "(5-fold, split by game). AUC 0.72; predicted goals match actual goals within 1%. Public models without tracking data land in a similar range.",
        "<b>Data problem found and removed:</b> shot-type labels are recorded inconsistently by arena. In Anaheim, 38% of shots for both teams are "
        "logged as snap shots vs 24% elsewhere, which would have made the Ducks look artificially weak against snap shots and tips. "
        "I excluded shot type from the model and dropped shot-type results. Location did not show this bias (slot share against was 44% at home and on the road).",
        "<b>Chance-source labels are heuristics.</b> The feed has no zone entries or passes. Rush = last logged event in the neutral/defensive zone "
        "within 10 s. At 5 s the rush share is too small to measure; at 5, 10 and 15 s the Ducks' rush excess is never distinguishable from zero.",
        "<b>On-ice attribution</b> reconstructs exactly five Ducks skaters on the ice for 99.7% of 5v5 seconds (worst game 97.8%).",
        "<b>Inference:</b> intervals resample only the Ducks' games (treated as independent) and hold the league average fixed, so they understate uncertainty. On/off is confounded by usage. I examined four zones, five chance sources, "
        "several splits and 20 players without correcting for multiple comparisons. 'Systemic, not personnel' rests on absence of evidence from low-powered on/off data.",
        "<b>Limits:</b> one season; no puck or player tracking, so I can show <i>where and when</i> chances arise but not <i>why</i>.",
    ], SM),
]

doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch, topMargin=0.65 * inch,
                        bottomMargin=0.8 * inch, title="Where the Ducks' 5v5 defense leaks", author=AUTHOR)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
