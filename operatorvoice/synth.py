"""Synthetic feedback from drivers, operators and car owners, with known themes.

The corpus is generated from sentence templates, so it is far cleaner than real feedback.
It exists to test the pipeline against a known answer. Use `load_csv` for real public posts."""

import random

import pandas as pd

# name: (label, software-addressable truth 0..1, posts, pain 0..1, subjects, issues, contexts)
THEMES = {
    "route_limits": (
        "Routes ignore vehicle height and weight",
        0.9,
        90,
        0.85,
        ["the navigation", "the route planner", "the map app", "route guidance"],
        [
            "ignores the height of my truck",
            "never checks weight limits",
            "sent us under a low bridge",
            "does not warn about tunnel height",
            "does not know my trailer dimensions",
            "chose a road we could not use",
        ],
        [
            "so I had to turn around",
            "again this week",
            "with a full load on board",
            "and the bridge sign was the first warning",
            "and the dispatcher had to reroute us",
        ],
    ),
    "charging_planning": (
        "Charging stops and range estimates are unreliable",
        0.9,
        70,
        0.8,
        ["the range estimate", "the charging planner", "the app", "the battery display"],
        [
            "drops much faster than shown in cold weather",
            "picked a charger that was busy",
            "ignores the load I carry",
            "is too optimistic after a long climb",
            "does not show live charger availability",
            "left me with range anxiety",
        ],
        [
            "so I waited an hour at the stop",
            "every winter",
            "on the motorway",
            "and the trip took much longer",
            "because it does not use battery temperature",
        ],
    ),
    "alert_overload": (
        "Too many alerts and notifications",
        0.95,
        60,
        0.6,
        ["the alerts", "the notifications", "the warning chimes", "the app"],
        [
            "beep for every minor thing",
            "come at night for things that need no action",
            "fill the screen while driving",
            "cannot be filtered or prioritised",
            "are so frequent that I mute them",
            "hide the warnings that matter",
        ],
        [
            "and it is distracting on the motorway",
            "so I miss the important ones",
            "and the settings are too limited",
            "every single trip",
            "and my colleagues turned them off",
        ],
    ),
    "pairing_drops": (
        "Phone connection and pairing drops",
        0.85,
        55,
        0.7,
        ["bluetooth pairing", "the phone connection", "the sync", "the app login"],
        [
            "drops every time I start the vehicle",
            "fails after every software update",
            "forgets the saved phone",
            "takes several tries to reconnect",
            "keeps logging me out",
            "loses the display connection mid trip",
        ],
        [
            "and I must pair it again",
            "which is annoying before a shift",
            "since the last update",
            "on every trip",
            "and support had no fix",
        ],
    ),
    "service_reminders": (
        "Maintenance reminders are wrong or missing",
        0.8,
        45,
        0.55,
        ["the service reminder", "the maintenance schedule", "the reminder in the app", "the service plan"],
        [
            "came after the due date",
            "does not count engine hours",
            "says service is due when it was done last week",
            "is missing for the oil change",
            "does not match the dealer plan",
            "never reminds me about the machine",
        ],
        [
            "so I track it in a notebook",
            "and the report shows something else",
            "on the excavator",
            "again this month",
            "and we nearly missed an inspection",
        ],
    ),
    "voice_control": (
        "Voice commands misunderstood",
        0.8,
        40,
        0.5,
        ["voice control", "the voice command", "the assistant", "speech recognition"],
        [
            "does not understand my accent",
            "calls the wrong contact",
            "fails with engine noise",
            "opened music instead of navigation",
            "only knows a few phrases",
            "hears nothing through my gloves",
        ],
        [
            "so I repeat it three times",
            "which is unsafe while driving",
            "in the cab",
            "every morning",
            "and I gave up using it",
        ],
    ),
    "fuel_reporting": (
        "Fuel and idling reports are hard to use",
        0.9,
        35,
        0.4,
        ["the fuel report", "the idling report", "the consumption dashboard", "the weekly report"],
        [
            "is a long table nobody reads",
            "arrives a week late",
            "does not split loaded and empty trips",
            "does not match the fuel card",
            "cannot be broken down by driver",
            "needs an export to make a chart",
        ],
        [
            "so I build the charts myself",
            "for the whole fleet",
            "and idling goes unnoticed",
            "every month",
            "for our managers",
        ],
    ),
    "display_glare": (
        "Display unreadable in sunlight",
        0.5,
        30,
        0.45,
        ["the display", "the screen in the cab", "the dashboard text", "the touch menu"],
        [
            "is unreadable in direct sun",
            "is too slow to adapt after a tunnel",
            "is tiny with sunglasses on",
            "reflects in the windscreen",
            "is hard to use with dirty gloves",
            "glares on low sun mornings",
        ],
        [
            "and I squint while driving",
            "on the early shift",
            "in every season",
            "so I turn it off",
            "and the buttons are too small",
        ],
    ),
    "seat_comfort": (
        "Seat comfort on long shifts",
        0.1,
        50,
        0.5,
        ["the driver seat", "the seat cushion", "the lumbar support", "the seat padding"],
        [
            "gets uncomfortable after a long shift",
            "is too hard on rough roads",
            "has worn down after two years",
            "squeaks and has no adjustment for tall drivers",
            "hurts my back",
            "makes my shoulders ache",
        ],
        ["after eight hours", "on long hauls", "even with the heating on", "and I need a pillow", "every week"],
    ),
    "road_noise": (
        "Cabin and tyre noise",
        0.05,
        45,
        0.4,
        ["the tyre noise", "the wind noise", "the cabin noise", "the door trim"],
        [
            "is loud on the motorway",
            "gets worse above eighty",
            "rattles every winter",
            "comes from the suspension on rough roads",
            "makes phone calls impossible",
            "leaks in around the window seal",
        ],
        [
            "and I get tired on long trips",
            "so I wear ear protection",
            "in the cab",
            "since the winter tyres",
            "and it is the same in the other trucks",
        ],
    ),
}
FILLER = [
    "honestly",
    "really",
    "again",
    "this week",
    "since the last update",
    "on every trip",
    "for months now",
    "my colleagues say the same",
    "very frustrating",
    "please fix this",
    "",
]
OFF_TOPIC = [
    "great vehicle overall",
    "delivery took a while",
    "bought it second hand",
    "nice colour",
    "dealer was friendly",
    "thanks for the quick reply",
    "ok product",
    "would buy again",
]
SOURCES = ["forum", "app_review", "survey", "dealer_note"]


def generate(seed: int = 1, noise_share: float = 0.1) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    others = list(THEMES)
    for key, (_label, _feas, n, pain, subjects, issues, contexts) in THEMES.items():
        for _ in range(n):
            text = f"{rng.choice(subjects)} {rng.choice(issues)}"
            if rng.random() < 0.7:
                text += " " + rng.choice(contexts)
            if rng.random() < 0.5:
                text = f"{rng.choice(FILLER)} {text}".strip()
            if rng.random() < 0.35:  # a loosely related remark about another topic
                o = THEMES[rng.choice([t for t in others if t != key])]
                text += ". " + f"{rng.choice(o[4])} {rng.choice(o[5])}"
            elif rng.random() < 0.15:
                text += ". " + rng.choice(OFF_TOPIC)
            rating = max(1, min(5, round(rng.gauss(3 - 2 * pain, 0.8) + 0.5)))
            rows.append({"text": text, "source": rng.choice(SOURCES), "rating": rating, "theme": key})
    n_noise = int(len(rows) * noise_share)
    for _ in range(n_noise):
        rows.append(
            {
                "text": rng.choice(OFF_TOPIC) + " " + rng.choice(FILLER),
                "source": rng.choice(SOURCES),
                "rating": rng.choice([3, 4, 5]),
                "theme": "none",
            }
        )
    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def ground_truth() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "theme": k,
                "label": v[0],
                "feasibility": v[1],
                "posts": v[2],
                "pain": v[3],
                "priority": v[2] * v[3] * v[1],
            }
            for k, v in THEMES.items()
        ]
    )
