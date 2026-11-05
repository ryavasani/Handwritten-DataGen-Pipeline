"""Synthetic Hindi text for plain handwritten legal documents (petitions,
applications, affidavits). Names and places are random combinations, not real
people or cases."""
from __future__ import annotations

import numpy as np

DIGITS = str.maketrans("0123456789", "०१२३४५६७८९")
GIVEN_M = "राम श्याम सुनील राजेश अनिल मनोज विकास सुरेश महेश दिनेश अजय विजय संजय रमेश प्रमोद अशोक राकेश मुकेश".split()
GIVEN_F = "सीता गीता पूजा सुनीता अनीता रेखा कविता ममता सरिता नीलम".split()
SURNAMES = "शर्मा वर्मा सिंह यादव गुप्ता कुमार पाल मिश्रा तिवारी चौहान पांडे जैन".split()
DISTRICTS = "लखनऊ कानपुर आगरा वाराणसी पटना जयपुर भोपाल इंदौर मेरठ प्रयागराज".split()
LOCALITIES = ["मोहल्ला", "ग्राम", "कॉलोनी", "नगर"]
PLACE_NAMES = ["रामनगर", "शिवपुरी", "गांधीनगर", "नेहरू नगर", "आदर्श नगर", "लक्ष्मी नगर"]


def _num(n: int) -> str:
    return str(n).translate(DIGITS)


def _date(rng) -> str:
    return f"{int(rng.integers(1, 29)):02d}/{int(rng.integers(1, 13)):02d}/{int(rng.integers(2019, 2025))}".translate(DIGITS)


def _name(rng) -> str:
    given = GIVEN_M if rng.random() < 0.75 else GIVEN_F
    return f"{rng.choice(given)} {rng.choice(SURNAMES)}"


def _address(rng) -> str:
    return f"{rng.choice(LOCALITIES)} {rng.choice(PLACE_NAMES)} {rng.choice(DISTRICTS)}"


TITLES = ["न्यायालय श्रीमान जिला न्यायाधीश महोदय {district}",
          "न्यायालय श्रीमान सिविल जज महोदय {district}",
          "न्यायालय श्रीमान उप जिलाधिकारी महोदय {district}",
          "प्रार्थना पत्र", "शपथ पत्र"]
SUBJECTS = ["भूमि विवाद के संबंध में", "उधार की रकम वापस न करने के संबंध में",
            "संपत्ति के बंटवारे के संबंध में", "जमानत हेतु प्रार्थना पत्र", "नाम संशोधन के संबंध में"]
OPENING = "निवेदन है कि प्रार्थी {name} पुत्र {father} निवासी {address} का स्थायी निवासी है।"
OATH = "मैं {name} पुत्र {father} उम्र {age} वर्ष निवासी {address} शपथपूर्वक बयान करता हूं कि"
FACTS = [
    "प्रार्थी की भूमि खसरा संख्या {num} ग्राम {village} में स्थित है।",
    "उक्त भूमि पर प्रार्थी का कब्जा पिछले {years} वर्षों से चला आ रहा है।",
    "दिनांक {date} को विपक्षी {opp} ने उक्त भूमि पर जबरन कब्जा करने का प्रयास किया।",
    "प्रार्थी ने इसकी सूचना स्थानीय थाने में दी परंतु कोई कार्यवाही नहीं हुई।",
    "विपक्षी द्वारा प्रार्थी को जान से मारने की धमकी दी जा रही है।",
    "प्रार्थी एक गरीब किसान है और उसके पास जीविका का कोई अन्य साधन नहीं है।",
    "प्रार्थी ने विपक्षी को रुपये {amount} उधार दिए थे जो आज तक वापस नहीं किए गए।",
    "कई बार मांगने पर भी विपक्षी टालमटोल करता रहा।",
    "गांव के कई लोग इस घटना के गवाह हैं।",
    "प्रार्थी के पिता की मृत्यु दिनांक {date} को हो गई थी।",
    "उनकी संपत्ति का बंटवारा अभी तक नहीं हुआ है।",
    "विपक्षी ने गलत तरीके से कागजात तैयार करवा लिए हैं।",
    "प्रार्थी ने कई बार अधिकारियों से शिकायत की परंतु कोई सुनवाई नहीं हुई।",
]
CLAUSES = [
    "यह कि प्रार्थी का इस मामले से संबंधित कोई अन्य वाद किसी न्यायालय में लंबित नहीं है।",
    "यह कि उपरोक्त सभी तथ्य प्रार्थी की जानकारी में सत्य एवं सही हैं।",
    "यह कि प्रार्थी न्यायालय के प्रत्येक आदेश का पालन करेगा।",
    "यह कि प्रार्थी जमानत की सभी शर्तों का पालन करेगा।",
]
PRAYERS = ["अतः श्रीमान जी से प्रार्थना है कि उचित आदेश पारित करने की कृपा करें।",
           "अतः श्रीमान जी से निवेदन है कि विपक्षी के विरुद्ध उचित कार्यवाही की जाए।"]
VERIFY = "मैं शपथकर्ता घोषणा करता हूं कि ऊपर लिखी सभी बातें मेरी जानकारी में सत्य हैं।"
MARGIN_NOTES = ["नकल प्राप्त की", "पेश हो", "दाखिल दफ्तर", "आदेश हेतु पेश", "सुनवाई दिनांक {date}"]


def sample_document(rng: np.random.Generator) -> dict:
    """Blocks of one document, in reading order. Each block is {"name",
    "text" or "sentences" (paragraphs), "align": left|center|right, "indent",
    "underline", "signature"}."""
    district = str(rng.choice(DISTRICTS))
    name, father = _name(rng), f"{rng.choice(GIVEN_M)} {_name(rng).split()[1]}"
    slots = {"district": district, "name": name, "father": father, "address": _address(rng),
             "age": _num(int(rng.integers(22, 70))), "date": _date(rng), "num": _num(int(rng.integers(11, 999))),
             "village": str(rng.choice(PLACE_NAMES)), "years": _num(int(rng.integers(5, 40))),
             "opp": _name(rng), "amount": _num(int(rng.integers(2, 90)) * 1000)}
    fill = lambda s: s.format(**slots)  # noqa: E731

    title = str(rng.choice(TITLES))
    affidavit = title == "शपथ पत्र"
    blocks = [{"name": "title", "text": fill(title), "align": "center", "underline": rng.random() < 0.6}]
    if not affidavit:
        blocks.append({"name": "subject", "text": "विषय: " + str(rng.choice(SUBJECTS)), "align": "left"})
        blocks.append({"name": "salutation", "text": "महोदय,", "align": "left"})
    facts = list(rng.choice(FACTS, size=int(rng.integers(9, len(FACTS) + 1)), replace=False))
    n = len(facts)  # more than fits; the page layout drops sentences to fill one sheet
    paras = [[OATH if affidavit else OPENING] + facts[: n // 3], facts[n // 3 : 2 * n // 3], facts[2 * n // 3 :]]
    if affidavit or rng.random() < 0.5:
        paras.append(list(rng.choice(CLAUSES, size=int(rng.integers(2, 4)), replace=False)))
    paras.append([VERIFY] if affidavit else [str(rng.choice(PRAYERS)), "प्रार्थी सदैव आभारी रहेगा।"])
    for i, p in enumerate(q for q in paras if q):
        blocks.append({"name": f"paragraph_{i + 1}", "sentences": [fill(s) for s in p], "align": "left",
                       "indent": True})
    blocks.append({"name": "signer", "text": "शपथकर्ता" if affidavit else "प्रार्थी", "align": "right"})
    blocks.append({"name": "signature", "text": name, "align": "right", "signature": True})
    blocks.append({"name": "date", "text": f"दिनांक {slots['date']}", "align": "left"})
    blocks.append({"name": "place", "text": f"स्थान {district}", "align": "left"})
    return {"blocks": blocks, "margin_note": fill(str(rng.choice(MARGIN_NOTES))), "slots": slots}
