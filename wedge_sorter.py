#!/usr/bin/env python3
"""
wedge_sorter.py - demo classifier for wedge bolts (standard / long / other).

Photograph one part at a time on a light, evenly lit surface (a white-screen
tablet or LED panel under the part is best). The script segments the part,
measures its shape (length, taper, holes, slots), and says which bucket it
would go to:  standard -> LEFT,  long -> RIGHT,  everything else -> MIDDLE.
Low confidence also goes to MIDDLE.

Install:   pip install opencv-python numpy scikit-learn
Workflow:
  python wedge_sorter.py selftest                     # check everything works (synthetic parts)
  python wedge_sorter.py capture --source 0           # save labeled photos from a camera
  python wedge_sorter.py train   --data dataset       # train + cross-validated accuracy
  python wedge_sorter.py eval    --data new_photos    # accuracy on photos the model never saw
  python wedge_sorter.py live    --source 0           # live camera with bucket overlay
  python wedge_sorter.py predict photo.jpg            # one image

Android phone as the camera: install an app such as "IP Webcam", start the server,
then use  --source http://PHONE_IP:8080/video   (phone and computer on the same Wi-Fi).
DroidCam / Iriun also work: they show up as a normal camera index (0, 1, 2...).
"""
import argparse
import os
import pickle
import sys
import time
from collections import Counter

import cv2
import numpy as np

CLASSES = ["standard", "long", "other"]
BUCKET = {"standard": "LEFT", "long": "RIGHT", "other": "MIDDLE"}
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
MODEL_PATH = "wedge_model.pkl"
MIN_AREA_FRAC = 0.002      # ignore blobs smaller than this fraction of the frame
CONF_THRESHOLD = 0.80      # below this -> "unsure" -> MIDDLE bucket


# ----------------------------------------------------------------------------
# Segmentation + feature extraction
# ----------------------------------------------------------------------------
def segment(img):
    """Return (part_mask, hole_mask) for the main part, or (None, None).

    Assumes a part that is darker than its background. Otsu thresholding on a
    blurred grayscale image means oil, paint and concrete dust barely matter
    because only the silhouette is used."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    h, w = bw.shape
    cnts, hier = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    best, best_area = None, 0
    for i, c in enumerate(cnts):
        if hier[0][i][3] != -1:          # skip holes
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        if x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1:
            continue                      # touches the frame edge: not a whole part
        a = cv2.contourArea(c)
        if a > best_area:
            best, best_area = i, a
    if best is None or best_area < MIN_AREA_FRAC * h * w:
        return None, None
    return cnts, (hier, best)


def features(img):
    """Return (feature_vector, debug_info) or (None, None) if no part found."""
    cnts, aux = segment(img)
    if cnts is None:
        return None, None
    hier, idx = aux
    c = cnts[idx]
    area = cv2.contourArea(c)

    # holes/slots = child contours of the part contour
    holes = []
    child = hier[0][idx][2]
    while child != -1:
        ha = cv2.contourArea(cnts[child])
        if ha > 0.003 * area:             # ignore specks
            holes.append(cnts[child])
        child = hier[0][child][0]

    # orientation via PCA of the silhouette points
    pts = c.reshape(-1, 2).astype(np.float32)
    mean, eig = cv2.PCACompute(pts, mean=None)
    axis = eig[0]
    perp = np.array([-axis[1], axis[0]])
    u = (pts - mean[0]) @ axis
    v = (pts - mean[0]) @ perp
    length = u.max() - u.min()
    width = v.max() - v.min()

    # width profile along the long axis (10 bins), flipped so the wider end is first
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, [c], -1, 255, -1)
    ys, xs = np.nonzero(mask)
    pu = (np.stack([xs, ys], 1) - mean[0]) @ axis
    pv = (np.stack([xs, ys], 1) - mean[0]) @ perp
    bins = np.clip(((pu - u.min()) / max(length, 1) * 10).astype(int), 0, 9)
    prof = np.zeros(10)
    for b in range(10):
        sel = pv[bins == b]
        prof[b] = (sel.max() - sel.min()) if len(sel) else 0
    if prof[:3].mean() < prof[7:].mean():
        prof = prof[::-1]
    prof = prof / max(prof.max(), 1)

    hull_area = cv2.contourArea(cv2.convexHull(c))
    hole_areas = sorted([cv2.contourArea(h) for h in holes], reverse=True)
    round_holes = slots = 0
    for h in holes:
        (_, _), (hw, hh), _ = cv2.minAreaRect(h)
        ar = max(hw, hh) / max(min(hw, hh), 1)
        if ar > 2.0:
            slots += 1
        else:
            round_holes += 1

    f = [
        length / max(width, 1),            # aspect ratio (scale independent)
        length / np.sqrt(area),            # elongation
        area / max(hull_area, 1),          # solidity
        len(holes), round_holes, slots,
        sum(hole_areas) / area,            # hole fraction
        (hole_areas[0] / area) if hole_areas else 0,
        length,                            # pixels: needs a fixed camera distance
    ] + list(prof)
    info = {"contour": c, "holes": holes, "length_px": length, "width_px": width,
            "n_holes": round_holes, "n_slots": slots}
    return np.array(f, np.float32), info


# ----------------------------------------------------------------------------
# Dataset / model
# ----------------------------------------------------------------------------
def load_dataset(root):
    X, y, paths = [], [], []
    skipped = 0
    for cls in CLASSES:
        d = os.path.join(root, cls)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.lower().endswith(IMG_EXT):
                continue
            p = os.path.join(d, fn)
            img = cv2.imread(p)
            f, _ = features(img) if img is not None else (None, None)
            if f is None:
                skipped += 1
                print(f"  skipped (no part found): {p}")
                continue
            X.append(f); y.append(cls); paths.append(p)
    return np.array(X), np.array(y), paths, skipped


def make_model():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=300, random_state=0, class_weight="balanced")


def report(y_true, y_pred, title):
    labels = CLASSES + ["unsure"]
    print(f"\n{title}")
    print(f"{'':>10}" + "".join(f"{l:>10}" for l in labels) + "   <- predicted")
    for t in CLASSES:
        row = Counter(p for tt, p in zip(y_true, y_pred) if tt == t)
        print(f"{t:>10}" + "".join(f"{row.get(l, 0):>10}" for l in labels))
    n = len(y_true)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    unsure = sum(p == "unsure" for p in y_pred)
    wrong = n - correct - unsure
    print(f"\n  correct: {correct}/{n} ({100*correct/max(n,1):.1f}%)   "
          f"unsure -> MIDDLE: {unsure}   confidently WRONG: {wrong}")
    # the number that matters for the machine: standard/long parts sent to the wrong
    # named bucket (an 'other' part in the middle bucket is always safe).
    bad = sum(1 for t, p in zip(y_true, y_pred)
              if BUCKET.get(p, "MIDDLE") != "MIDDLE" and BUCKET[p] != BUCKET[t])
    print(f"  parts that would land in the WRONG left/right bucket: {bad}")


def predict_features(model, f):
    proba = model.predict_proba(f.reshape(1, -1))[0]
    k = int(np.argmax(proba))
    cls, conf = model.classes_[k], float(proba[k])
    if conf < CONF_THRESHOLD:
        return "unsure", conf
    return cls, conf


def cmd_train(a):
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    X, y, _, _ = load_dataset(a.data)
    if len(set(y)) < 2:
        sys.exit("Need photos in at least two class folders: " + ", ".join(CLASSES))
    counts = Counter(y)
    print("Photos per class:", dict(counts))
    folds = max(2, min(5, min(counts.values())))
    model = make_model()
    proba = cross_val_predict(model, X, y, cv=StratifiedKFold(folds, shuffle=True, random_state=0),
                              method="predict_proba")
    classes = sorted(set(y))
    pred = []
    for p in proba:
        k = int(np.argmax(p))
        pred.append(classes[k] if p[k] >= CONF_THRESHOLD else "unsure")
    report(list(y), pred, f"{folds}-fold cross-validated accuracy (each photo scored by a model that never saw it)")
    model.fit(X, y)
    with open(a.model, "wb") as fh:
        pickle.dump(model, fh)
    print(f"\nSaved model to {a.model}")


def load_model(path):
    if not os.path.exists(path):
        sys.exit(f"No model at {path}. Run 'train' first.")
    with open(path, "rb") as fh:
        return pickle.load(fh)


def cmd_eval(a):
    model = load_model(a.model)
    X, y, paths, skipped = load_dataset(a.data)
    pred = []
    for f, p, t in zip(X, paths, y):
        cls, conf = predict_features(model, f)
        pred.append(cls)
        if cls != t:
            print(f"  miss: {p}  true={t} predicted={cls} ({conf:.2f})")
    report(list(y), pred, "Held-out accuracy")
    if skipped:
        print(f"  ({skipped} photos skipped: no part found)")


# ----------------------------------------------------------------------------
# Display helpers, predict, live, capture
# ----------------------------------------------------------------------------
def annotate(img, info, cls, conf):
    out = img.copy()
    if info is None:
        cv2.putText(out, "no part found", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return out
    color = {"standard": (0, 160, 0), "long": (200, 100, 0), "other": (0, 140, 255),
             "unsure": (0, 0, 255)}[cls]
    cv2.drawContours(out, [info["contour"]], -1, color, 3)
    cv2.drawContours(out, info["holes"], -1, (255, 0, 255), 2)
    bucket = BUCKET.get(cls, "MIDDLE")
    txt = f"{cls.upper()}  ->  {bucket}   ({conf:.0%})  holes={info['n_holes']} slots={info['n_slots']}"
    cv2.rectangle(out, (0, 0), (out.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(out, txt, (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return out


def cmd_predict(a):
    model = load_model(a.model)
    for p in a.images:
        img = cv2.imread(p)
        if img is None:
            print(f"{p}: could not read"); continue
        f, info = features(img)
        if f is None:
            print(f"{p}: no part found -> MIDDLE (treat as unsure)"); continue
        cls, conf = predict_features(model, f)
        print(f"{p}: {cls} ({conf:.0%}) -> {BUCKET.get(cls, 'MIDDLE')}")
        if a.save:
            out = os.path.splitext(p)[0] + "_result.jpg"
            cv2.imwrite(out, annotate(img, info, cls, conf))
            print(f"  wrote {out}")


def open_source(src):
    cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
    if not cap.isOpened():
        sys.exit(f"Could not open camera/stream: {src}")
    return cap


def cmd_live(a):
    model = load_model(a.model)
    cap = open_source(a.source)
    print("Live mode. Place one part in view. Keys: q quit, s save frame.")
    votes = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        f, info = features(frame)
        if f is None:
            votes.clear()
            shown = annotate(frame, None, "unsure", 0)
        else:
            cls, conf = predict_features(model, f)
            votes = (votes + [cls])[-5:]           # smooth over the last 5 frames
            top, n = Counter(votes).most_common(1)[0]
            if n < 4:                              # not stable yet
                top = "unsure"
            shown = annotate(frame, info, top, conf)
        cv2.imshow("wedge sorter", shown)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s"):
            cv2.imwrite(f"frame_{int(time.time())}.jpg", frame)
    cap.release()
    cv2.destroyAllWindows()


def cmd_capture(a):
    cap = open_source(a.source)
    keys = {ord("1"): "standard", ord("2"): "long", ord("3"): "other"}
    for c in CLASSES:
        os.makedirs(os.path.join(a.data, c), exist_ok=True)
    print("Keys: 1=standard  2=long  3=other (adjustable/base tie/anything else)  q=quit")
    n = Counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        f, info = features(frame)
        view = annotate(frame, info, "other", 0) if info else frame
        counts = "  ".join(f"{c}:{sum(1 for _ in os.listdir(os.path.join(a.data, c)))}" for c in CLASSES)
        cv2.putText(view, counts, (10, view.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.imshow("capture", view)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k in keys:
            if f is None:
                print("no part found in frame, not saved"); continue
            name = f"{int(time.time()*1000)}.jpg"
            cv2.imwrite(os.path.join(a.data, keys[k], name), frame)
            n[keys[k]] += 1
    cap.release()
    cv2.destroyAllWindows()
    print("Saved this session:", dict(n))


# ----------------------------------------------------------------------------
# Synthetic self-test (so the pipeline can be verified without any photos)
# ----------------------------------------------------------------------------
def _poly(kind, rng):
    """Part outlines roughly like the reference photo, in a local frame (units ~ px)."""
    j = lambda s=1.0: rng.uniform(-2, 2) * s
    if kind == "standard":      # ~ medium length, tapering to a point, slot + hole
        L = 300 + j(6)
        outer = [(0, -26), (18, -26), (18, -16), (L, -3), (L, 3), (18, 16), (18, 26), (0, 26)]
        holes = [("slot", 80, 0, 70, 12), ("circ", 160, 0, 11, 11)]
    elif kind == "long":        # longer, nearly parallel, flat taper at the tip, 2 holes + slot
        L = 430 + j(8)
        outer = [(0, -30), (20, -30), (20, -17), (L - 40, -17), (L, -6), (L, 6), (L - 40, 17), (20, 17), (20, 30), (0, 30)]
        holes = [("circ", 60, 0, 11, 11), ("circ", 100, 0, 11, 11), ("slot", L - 100, 0, 70, 12)]
    elif kind == "adjustable":  # long, round ends, 5 holes + 2 slots
        L = 440 + j(8)
        outer = [(0, -20), (L - 20, -20), (L, -8), (L, 8), (L - 20, 20), (0, 20)]
        holes = [("slot", 40, 0, 60, 10), ("slot", L - 60, 0, 60, 10)] + \
                [("circ", 130 + 45 * i, 0, 9, 9) for i in range(5)]
    else:                       # base tie: small, stubby, one hole
        L = 110 + j(4)
        outer = [(0, -30), (22, -30), (22, -12), (L, 8), (L, 22), (22, 22), (22, 30), (0, 30)]
        holes = [("circ", L - 30, 8, 8, 8)]
    return np.array(outer, np.float32), holes


def synth_image(kind, rng, size=(720, 960)):
    h, w = size
    bg = int(rng.uniform(215, 250))
    img = np.full((h, w, 3), bg, np.uint8)
    img = (img + rng.normal(0, 3, img.shape)).clip(0, 255).astype(np.uint8)
    outer, holes = _poly(kind, rng)
    scale = rng.uniform(0.97, 1.03)
    ang = rng.uniform(0, 2 * np.pi)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    cx, cy = w / 2 + rng.uniform(-60, 60), h / 2 + rng.uniform(-40, 40)
    ctr = outer.mean(0)
    T = lambda p: ((np.asarray(p, np.float32) - ctr) * scale) @ R.T + [cx, cy]
    tone = int(rng.uniform(35, 90))
    cv2.fillPoly(img, [T(outer).astype(np.int32)], (tone, tone, tone))
    for kind_, x, y, a, b in holes:
        c = T([x, y])
        if kind_ == "circ":
            cv2.circle(img, tuple(c.astype(int)), int(a * scale), (bg, bg, bg), -1)
        else:
            ends = [T([x - a / 2, y]), T([x + a / 2, y])]
            cv2.line(img, tuple(ends[0].astype(int)), tuple(ends[1].astype(int)), (bg, bg, bg), int(b * 2 * scale))
    # "dirt": speckles and smears on the part, plus a pink paint blotch
    for _ in range(25):
        p = T(outer[rng.integers(len(outer))] * rng.uniform(0.2, 0.9) + ctr * 0.3)
        cv2.circle(img, tuple(p.astype(int)), int(rng.uniform(1, 4)),
                   (int(tone + 40), int(tone + 40), int(tone + 40)), -1)
    if rng.random() < 0.5:
        p = T(outer.mean(0) + rng.uniform(-20, 20, 2))
        cv2.circle(img, tuple(p.astype(int)), int(rng.uniform(6, 14)), (120, 60, 220), -1)
    return cv2.GaussianBlur(img, (3, 3), 0)


def cmd_selftest(a):
    from sklearn.model_selection import train_test_split
    rng = np.random.default_rng(1)
    kinds = {"standard": "standard", "long": "long", "adjustable": "other", "base_tie": "other"}
    X, y, noparts = [], [], 0
    for kind, label in kinds.items():
        n = 120 if label == "other" else 100
        for i in range(n):
            f, _ = features(synth_image(kind if kind != "base_tie" else "base", rng))
            if f is None:
                noparts += 1; continue
            X.append(f); y.append(label)
    X, y = np.array(X), np.array(y)
    print(f"Generated {len(y)} synthetic parts ({noparts} unreadable). Classes: {dict(Counter(y))}")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
    model = make_model().fit(Xtr, ytr)
    pred = [predict_features(model, f)[0] for f in Xte]
    report(list(yte), pred, "Synthetic self-test on held-out parts")
    # leave a sample result image so the overlay can be checked by eye
    img = synth_image("long", rng)
    f, info = features(img)
    cls, conf = predict_features(model, f)
    out = a.out or "selftest_example.jpg"
    cv2.imwrite(out, annotate(img, info, cls, conf))
    print(f"\nExample annotated image: {out}")
    print("Note: synthetic parts are easy. Real photos of dirty parts are the real test.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("selftest"); p.add_argument("--out"); p.set_defaults(fn=cmd_selftest)
    p = sub.add_parser("capture"); p.add_argument("--source", default="0"); p.add_argument("--data", default="dataset"); p.set_defaults(fn=cmd_capture)
    p = sub.add_parser("train"); p.add_argument("--data", default="dataset"); p.add_argument("--model", default=MODEL_PATH); p.set_defaults(fn=cmd_train)
    p = sub.add_parser("eval"); p.add_argument("--data", required=True); p.add_argument("--model", default=MODEL_PATH); p.set_defaults(fn=cmd_eval)
    p = sub.add_parser("live"); p.add_argument("--source", default="0"); p.add_argument("--model", default=MODEL_PATH); p.set_defaults(fn=cmd_live)
    p = sub.add_parser("predict"); p.add_argument("images", nargs="+"); p.add_argument("--model", default=MODEL_PATH); p.add_argument("--save", action="store_true"); p.set_defaults(fn=cmd_predict)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
