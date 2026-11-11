"""Two-way choice tasks with no single correct answer.

Both tasks are built so that either answer is defensible. The experiment needs
items where the model's choice can be flipped and then justified; there is no
ground truth, only the model's own natural choice.

* sentiment: a short review that mixes praise and complaints -> Positive / Negative
* choice:    two options with a trade-off (cheaper vs. better)  -> A / B
"""

from __future__ import annotations

import random

import pandas as pd

PRODUCTS = ["blender", "laptop", "backpack", "phone", "coffee maker", "desk lamp", "headphone set",
            "rice cooker", "office chair", "water bottle", "smartwatch", "e-reader", "air fryer",
            "bicycle", "tent", "keyboard", "camera", "vacuum cleaner", "speaker", "kettle"]

POS = ["it looks great on my desk", "setup took less than five minutes", "the build quality feels solid",
       "it is very quiet", "the battery easily lasts all day", "it was a good price",
       "customer support answered quickly", "it is light and easy to carry", "the controls are intuitive",
       "it arrived a day early", "the colour is exactly as pictured", "it does the main job well"]

NEG = ["the manual is confusing", "it is heavier than I expected", "the cable is too short",
       "it gets warm after an hour", "one button already feels loose", "the packaging was excessive",
       "it is louder than advertised", "the app keeps asking for updates", "it scratches easily",
       "the lid is hard to open", "delivery took two weeks", "it only comes in one size"]

REVIEW_FORMS = [
    "I bought this {p} last month. {P1}, and {p2}. However, {n1}, and {n2}.",
    "This {p} has mixed results for me. {N1}, and {n2}. On the other hand, {p1}, and {p2}.",
    "Honest review of the {p}: {p1}, but {n1}. Also, {p2}, though {n2}.",
    "Two weeks with this {p}. {N1}. {P1}. {N2}. {P2}.",
]

DOMAINS = {
    "apartment": [("rent", "Rs {} per month", (14000, 30000)), ("commute", "{} minutes to work", (10, 60)),
                  ("size", "{} square feet", (450, 1100))],
    "laptop": [("price", "Rs {}", (45000, 110000)), ("battery", "{} hours of battery", (5, 16)),
               ("weight", "{} kg", (1.1, 2.4))],
    "job offer": [("salary", "Rs {} lakh per year", (6, 24)), ("hours", "{} hours a week", (38, 58)),
                  ("commute", "{} minutes away", (10, 75))],
    "phone plan": [("price", "Rs {} per month", (199, 899)), ("data", "{} GB of data per day", (1, 4)),
                   ("contract", "a {}-month contract", (1, 24))],
}

CHOICE_FORM = ("I need to pick a new {d}. Option A: {a}. Option B: {b}. "
               "Which option should I choose, A or B?")


def _cap(s):
    return s[0].upper() + s[1:]


def _sentiment(rng, n):
    rows = []
    for i in range(n):
        p = rng.choice(PRODUCTS)
        p1, p2 = rng.sample(POS, 2)
        n1, n2 = rng.sample(NEG, 2)
        form = REVIEW_FORMS[i % len(REVIEW_FORMS)]
        text = form.format(p=p, p1=p1, p2=p2, n1=n1, n2=n2, P1=_cap(p1), P2=_cap(p2), N1=_cap(n1), N2=_cap(n2))
        rows.append({"task": "sentiment", "item": text,
                     "question": f'Here is a product review:\n"{text}"\nIs this review Positive or Negative overall?',
                     "options": "Positive|Negative"})
    return rows


def _fmt(v):
    return f"{v:.1f}" if isinstance(v, float) else f"{v:,}"


def _choice(rng, n):
    rows = []
    names = list(DOMAINS)
    for i in range(n):
        d = names[i % len(names)]
        attrs = DOMAINS[d]
        vals_a, vals_b = [], []
        # values drawn independently per option, so most items involve a trade-off
        for j, (_, fmt, (lo, hi)) in enumerate(attrs):
            if isinstance(lo, float):
                x, y = round(rng.uniform(lo, hi), 1), round(rng.uniform(lo, hi), 1)
            else:
                x, y = rng.randint(lo, hi), rng.randint(lo, hi)
            vals_a.append(x)
            vals_b.append(y)
        a = ", ".join(fmt.format(_fmt(v)) for (_, fmt, _), v in zip(attrs, vals_a))
        b = ", ".join(fmt.format(_fmt(v)) for (_, fmt, _), v in zip(attrs, vals_b))
        rows.append({"task": "choice", "item": f"{d}: A = {a}; B = {b}",
                     "question": CHOICE_FORM.format(d=d, a=a, b=b), "options": "A|B"})
    return rows


def build_tasks(n_per_task=300, seed=0) -> pd.DataFrame:
    rng = random.Random(seed)
    df = pd.DataFrame(_sentiment(rng, n_per_task) + _choice(rng, n_per_task))
    df.insert(0, "item_id", range(len(df)))
    # half the items calibrate the steering vector, half are used for the experiment
    df["split"] = ["calib" if i % 2 == 0 else "test" for i in range(len(df))]
    return df


if __name__ == "__main__":
    d = build_tasks(4)
    for q in d.question:
        print(q, "\n")
