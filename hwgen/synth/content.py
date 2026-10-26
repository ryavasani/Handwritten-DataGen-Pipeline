"""Synthetic Hindi field values for FIR forms. Names are random combinations
of common given names and surnames, not real people; places are generic."""
from __future__ import annotations

import numpy as np

DIGITS = str.maketrans("0123456789", "०१२३४५६७८९")

GIVEN_M = "राम श्याम सुनील राजेश अनिल मनोज विकास सुरेश महेश दिनेश अजय विजय संजय रमेश प्रमोद अशोक राकेश मुकेश".split()
GIVEN_F = "सीता गीता पूजा सुनीता अनीता रेखा कविता ममता सरिता नीलम".split()
SURNAMES = "शर्मा वर्मा सिंह यादव गुप्ता कुमार पाल मिश्रा तिवारी चौहान पांडे जैन".split()
DISTRICTS = "लखनऊ कानपुर आगरा वाराणसी पटना जयपुर भोपाल इंदौर मेरठ प्रयागराज".split()
STATIONS = ["कोतवाली", "सदर", "सिविल लाइंस", "कैंट", "नगर", "शहर कोतवाली"]
DAYS = "सोमवार मंगलवार बुधवार गुरुवार शुक्रवार शनिवार रविवार".split()
DIRECTIONS = ["उत्तर", "दक्षिण", "पूर्व", "पश्चिम"]
LOCALITIES = ["मोहल्ला", "ग्राम", "कॉलोनी", "नगर"]
PLACE_NAMES = ["रामनगर", "शिवपुरी", "गांधीनगर", "नेहरू नगर", "आदर्श नगर", "लक्ष्मी नगर"]
ACTS = ["भा.दं.सं.", "भारतीय दंड संहिता", "भारतीय न्याय संहिता"]
SECTIONS = ["३७९", "३८०", "४२०", "३२३/५०४/५०६", "४२०/४०६", "३७९/४११", "३५४", "४५७/३८०", "३०७", "२७९/३३८"]
RANKS = ["उपनिरीक्षक", "निरीक्षक", "थाना प्रभारी", "हेड कांस्टेबल"]
PROPERTY = ["एक मोबाइल फोन", "नकद रुपये ५०००", "एक मोटरसाइकिल", "सोने की चेन", "एक साइकिल", "पर्स और नकद रुपये"]
DELAY = ["कोई विलंब नहीं", "विलंब नहीं हुआ", "रात होने के कारण"]
ACTIONS = ["मुकदमा दर्ज कर विवेचना प्रारंभ की गई।",
           "प्रथम सूचना रिपोर्ट दर्ज कर जांच शुरू की गई।",
           "मुकदमा दर्ज कर उपनिरीक्षक को विवेचना सौंपी गई।"]
NARRATIVE = [
    "निवेदन है कि दिनांक {date} को समय करीब {time} बजे मेरे घर से {property} चोरी हो गया।",
    "मैं {name} पुत्र {father} निवासी {address} का रहने वाला हूं।",
    "मैंने आसपास काफी तलाश की परंतु कुछ पता नहीं चला।",
    "मुझे शक है कि यह काम किसी अज्ञात व्यक्ति ने किया है।",
    "अतः श्रीमान जी से निवेदन है कि रिपोर्ट दर्ज कर उचित कार्यवाही करें।",
    "घटना के समय मैं अपने परिवार के साथ बाहर गया था।",
    "वापस आकर देखा तो घर का ताला टूटा हुआ था।",
    "दिनांक {date} को रास्ते में कुछ लोगों ने मेरे साथ मारपीट की।",
]


def _num(n: int) -> str:
    return str(n).translate(DIGITS)


def _date(rng) -> str:
    return f"{int(rng.integers(1, 29)):02d}/{int(rng.integers(1, 13)):02d}/{int(rng.integers(2019, 2025))}".translate(DIGITS)


def _name(rng) -> str:
    given = GIVEN_M if rng.random() < 0.75 else GIVEN_F
    return f"{rng.choice(given)} {rng.choice(SURNAMES)}"


def _address(rng) -> str:
    return f"{rng.choice(LOCALITIES)} {rng.choice(PLACE_NAMES)} {rng.choice(DISTRICTS)}"


def sample_values(rng: np.random.Generator) -> dict[str, str]:
    """One consistent set of values for every field of the form."""
    date = _date(rng)
    name, father = _name(rng), f"{rng.choice(GIVEN_M)} {rng.choice(SURNAMES)}"
    address = _address(rng)
    time = f"{int(rng.integers(1, 13))}:{int(rng.choice([0, 15, 30, 45])):02d}".translate(DIGITS)
    prop = str(rng.choice(PROPERTY))
    slots = {"date": date, "time": time, "property": prop, "name": name, "father": father, "address": address}
    k = int(rng.integers(3, 6))
    picks = [NARRATIVE[0], NARRATIVE[1]] + list(rng.choice(NARRATIVE[2:], size=k - 2, replace=False))
    year = date.split("/")[-1]
    return {
        "district": str(rng.choice(DISTRICTS)),
        "station": str(rng.choice(STATIONS)),
        "year": year,
        "fir_no": f"{_num(int(rng.integers(1, 999)))}/{year}",
        "fir_date": date,
        "act": str(rng.choice(ACTS)),
        "sections": str(rng.choice(SECTIONS)),
        "incident_day": str(rng.choice(DAYS)),
        "incident_date": date,
        "incident_time": f"{time} बजे",
        "received_date": date,
        "received_time": f"{_num(int(rng.integers(1, 13)))}:३० बजे",
        "direction": f"{rng.choice(DIRECTIONS)} {_num(int(rng.integers(1, 9)))} किमी",
        "place": address,
        "complainant": name,
        "father": father,
        "birth_year": _num(int(rng.integers(1955, 2004))),
        "nationality": "भारतीय",
        "phone": "".join(_num(int(d)) for d in rng.integers(0, 10, 10)),
        "address": address,
        "accused": "अज्ञात" if rng.random() < 0.6 else f"{_name(rng)} पुत्र {rng.choice(GIVEN_M)} निवासी {_address(rng)}",
        "delay": str(rng.choice(DELAY)),
        "property": prop,
        "narrative": " ".join(p.format(**slots) for p in picks),
        "action": str(rng.choice(ACTIONS)),
        "complainant_signature": name.split()[0],
        "officer_signature": str(rng.choice(GIVEN_M)),
        "officer_name": _name(rng),
        "officer_rank": str(rng.choice(RANKS)),
    }
