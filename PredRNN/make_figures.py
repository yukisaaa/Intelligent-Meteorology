"""Create GitHub-ready plots from the packaged training metrics and predictions."""
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
OUT = ROOT / "outputs"
FIG.mkdir(exist_ok=True)

REGULAR = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

def font(size, bold=False):
    path = BOLD if bold else REGULAR
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()

def text(draw, xy, value, size=22, fill="#263449", bold=False, anchor=None):
    draw.text(xy, str(value), font=font(size, bold), fill=fill, anchor=anchor)

def plot_curve(draw, box, title, values, color, fmt, ymin=None, ymax=None):
    x0, y0, x1, y1 = box
    left, right, top, bottom = x0 + 75, x1 - 24, y0 + 62, y1 - 62
    text(draw, (x0 + 8, y0 + 8), title, 27, "#17263C", True)
    lo = min(values) if ymin is None else ymin
    hi = max(values) if ymax is None else ymax
    span = max(hi - lo, 1e-8)
    lo -= span * 0.12
    hi += span * 0.12
    for i in range(5):
        yy = bottom - (bottom - top) * i / 4
        draw.line((left, yy, right, yy), fill="#E2E8F0", width=2)
        value = lo + (hi - lo) * i / 4
        text(draw, (left - 12, yy), fmt(value), 17, "#66758A", anchor="rm")
    draw.line((left, top, left, bottom), fill="#738096", width=2)
    draw.line((left, bottom, right, bottom), fill="#738096", width=2)
    n = len(values)
    points = []
    for i, value in enumerate(values):
        xx = left if n == 1 else left + (right - left) * i / (n - 1)
        yy = bottom - (value - lo) / (hi - lo) * (bottom - top)
        points.append((xx, yy))
        text(draw, (xx, bottom + 15), i + 1, 16, "#66758A", anchor="mt")
    if len(points) > 1:
        draw.line(points, fill=color, width=5, joint="curve")
    for point in points:
        r = 7
        draw.ellipse((point[0] - r, point[1] - r, point[0] + r, point[1] + r), fill=color, outline="white", width=2)
    text(draw, ((left + right) / 2, y1 - 20), "Epoch", 18, "#66758A", anchor="mm")

def training_plot():
    metrics = json.loads((OUT / "training_metrics.json").read_text())
    epochs = [item["epoch"] for item in metrics]
    train = [item["train_scaled_mse"] for item in metrics]
    val = [item["validation_rmse"] for item in metrics]
    best_i = min(range(len(val)), key=val.__getitem__)
    image = Image.new("RGB", (1680, 960), "#F6F8FB")
    draw = ImageDraw.Draw(image)
    text(draw, (75, 48), "PredRNN · CIKM2017", 42, "#142338", True)
    text(draw, (77, 110), f"Training history · 10 epochs · best validation RMSE {val[best_i]:.3f} mm (epoch {epochs[best_i]})", 22, "#556579")
    plot_curve(draw, (60, 185, 815, 885), "Training loss (scaled MSE)", train, "#3973D6", lambda x: f"{x:.3f}")
    plot_curve(draw, (865, 185, 1620, 885), "Validation RMSE (mm)", val, "#18A27A", lambda x: f"{x:.1f}")
    image.save(FIG / "training_curves.png", optimize=True)

def read_predictions(name):
    with (OUT / f"{name}_predictions.csv").open(newline="") as f:
        return [float(row["predicted_rain"]) for row in csv.DictReader(f)]

def prediction_histogram():
    a, b = read_predictions("testA"), read_predictions("testB")
    bins = [i * 1.25 for i in range(21)]
    counts_a = [0] * 20
    counts_b = [0] * 20
    for vals, counts in ((a, counts_a), (b, counts_b)):
        for value in vals:
            i = min(int(value / 1.25), 19)
            counts[i] += 1
    ymax = max(max(counts_a), max(counts_b)) * 1.16
    image = Image.new("RGB", (1500, 920), "#F6F8FB")
    draw = ImageDraw.Draw(image)
    text(draw, (80, 48), "PredRNN · Test prediction distribution", 40, "#142338", True)
    text(draw, (82, 108), "Predicted rainfall amount (mm) · 2,000 records per split · test labels are unavailable", 21, "#556579")
    left, right, top, bottom = 150, 1410, 205, 760
    for i in range(6):
        yy = bottom - (bottom - top) * i / 5
        val = ymax * i / 5
        draw.line((left, yy, right, yy), fill="#E2E8F0", width=2)
        text(draw, (left - 18, yy), f"{val:.0f}", 17, "#66758A", anchor="rm")
    draw.line((left, top, left, bottom), fill="#738096", width=2)
    draw.line((left, bottom, right, bottom), fill="#738096", width=2)
    step = (right - left) / 20
    for i in range(20):
        center = left + (i + .5) * step
        h_a = (bottom - top) * counts_a[i] / ymax
        h_b = (bottom - top) * counts_b[i] / ymax
        w = step * .36
        draw.rectangle((center - w - 2, bottom - h_a, center - 2, bottom), fill="#3973D6")
        draw.rectangle((center + 2, bottom - h_b, center + w + 2, bottom), fill="#18A27A")
    for val in range(0, 26, 5):
        x = left + (right - left) * val / 25
        text(draw, (x, bottom + 25), val, 17, "#66758A", anchor="mt")
    text(draw, ((left + right) / 2, bottom + 65), "Predicted rainfall (mm)", 20, "#556579", anchor="mm")
    draw.rectangle((930, 145, 958, 169), fill="#3973D6")
    text(draw, (970, 157), "TestA", 19, "#263449", anchor="lm")
    draw.rectangle((1060, 145, 1088, 169), fill="#18A27A")
    text(draw, (1100, 157), "TestB", 19, "#263449", anchor="lm")
    image.save(FIG / "prediction_distribution.png", optimize=True)

if __name__ == "__main__":
    training_plot()
    prediction_histogram()
    print(f"Wrote figures to {FIG}")
