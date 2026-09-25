"""Generate a small structured sample PDF so Phase 1 can be tested offline."""
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak

out = Path("data/raw_pdfs/sample_handbook.pdf")
out.parent.mkdir(parents=True, exist_ok=True)
styles = getSampleStyleSheet()
doc = SimpleDocTemplate(str(out), pagesize=letter)
story = []

def h(t): story.append(Paragraph(t, styles["Heading1"])); story.append(Spacer(1, 8))
def h2(t): story.append(Paragraph(t, styles["Heading2"])); story.append(Spacer(1, 6))
def p(t): story.append(Paragraph(t, styles["BodyText"])); story.append(Spacer(1, 6))

h("Widget Corp Field Service Handbook")
h2("1. Safety Procedures")
p("Before servicing any unit, disconnect mains power and verify zero voltage "
  "with a calibrated meter. Lockout-tagout is mandatory. Personal protective "
  "equipment includes insulated gloves rated to 1000V and safety eyewear.")
p("Capacitor banks may retain charge for up to five minutes after disconnection. "
  "Always discharge through the approved bleed resistor before touching terminals.")
h2("2. Routine Maintenance")
p("The Model X compressor requires oil inspection every 500 operating hours. "
  "Use only ISO VG 68 synthetic lubricant. Overfilling causes seal failure and "
  "voids the warranty. Record the oil level and any top-up in the service log.")
story.append(PageBreak())
h2("3. Troubleshooting")
p("If the unit fails to start, first confirm the supply breaker is closed and "
  "the control fuse F1 is intact. A blinking amber LED indicates a low-pressure "
  "lockout; check refrigerant charge and the high-side sensor before resetting.")
p("Error code E14 corresponds to a communication fault between the main board "
  "and the display module. Reseat the ribbon cable; if the fault persists, "
  "replace the display harness part number 55-2210.")
h2("4. Parts Reference")
p("Common spare parts: air filter 41-0087, oil filter 41-0091, contactor "
  "22-5540, and display harness 55-2210. Order through the regional depot with "
  "48-hour lead time for non-stock items.")

doc.build(story)
print(f"wrote {out} ({out.stat().st_size} bytes)")
