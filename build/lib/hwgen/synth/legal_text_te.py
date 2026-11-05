"""Synthetic Telugu text for plain handwritten legal documents (petitions to a
court, complaints to the police, affidavits), the Telugu counterpart of
legal_text.py. Names and places are random combinations, not real people or
cases. Dates and numbers use Latin digits, as handwritten Telugu papers do.

The sample library is small, so `ok` (can the bank write this text?) filters
the slot values: names, places and sentences are chosen among those it can
write, and the document builder drops any sentence that still fails.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

GIVEN_M = ("రాము శ్రీనివాస్ వెంకటేశ్వర్లు సురేష్ రమేష్ కృష్ణ నాగరాజు రవి సత్యనారాయణ మహేష్ "
           "రాజు గోపాల్ శ్రీను సాయి కుమార్ నరసింహ రామారావు ప్రసాద్ మోహన్ బాబు").split()
FAMILY = ("కొత్తపల్లి పెద్దిరెడ్డి బండారు చల్లా కోట మద్దాల వేముల పొట్లూరి యర్రా గుండు "
          "కొండా పాలడుగు నల్లూరి మన్నె బొమ్మ దాసరి").split()
DISTRICTS = "గుంటూరు విజయవాడ నెల్లూరు కర్నూలు ఒంగోలు కాకినాడ వరంగల్ ఖమ్మం కరీంనగర్ నల్గొండ అనంతపురం కడప".split()
VILLAGES = "రామాపురం కొత్తపేట శివాపురం లక్ష్మీపురం గాంధీనగర్ చింతలపూడి పెదపాడు నందిగామ మాచవరం తాడేపల్లి".split()

TITLES = ["గౌరవనీయులైన జిల్లా న్యాయమూర్తి గారి సన్నిధికి {district}",
          "గౌరవనీయులైన సివిల్ జడ్జి గారి న్యాయస్థానము {district}",
          "శ్రీయుత పోలీస్ ఇన్స్పెక్టర్ గారికి {district}",
          "అర్జీ", "ప్రమాణ పత్రము"]
SUBJECTS = ["భూమి వివాదం గురించి", "అప్పు తిరిగి చెల్లించకపోవడం గురించి", "ఆస్తి పంపకం గురించి",
            "బెయిల్ మంజూరు కొరకు అర్జీ", "పేరు సవరణ గురించి", "దొంగతనం గురించి ఫిర్యాదు"]
OPENING = "{name} తండ్రి {father} నివాసం {address} అను నేను ఈ అర్జీ ద్వారా మనవి చేయునది ఏమనగా"
OATH = ("{name} తండ్రి {father} వయస్సు {age} సంవత్సరాలు నివాసం {address} అను నేను "
        "ప్రమాణపూర్వకంగా తెలియజేయునది ఏమనగా")
FACTS = [
    "అర్జీదారుని భూమి సర్వే నంబరు {num} {village} గ్రామంలో ఉన్నది.",
    "ఆ భూమి గత {years} సంవత్సరాలుగా అర్జీదారుని అనుభవంలో ఉన్నది.",
    "తేదీ {date} న ప్రతివాది {opp} ఆ భూమిని బలవంతంగా ఆక్రమించడానికి ప్రయత్నించాడు.",
    "అర్జీదారు ఈ విషయాన్ని పోలీసులకు తెలియజేసినా ఎలాంటి చర్య తీసుకోలేదు.",
    "ప్రతివాది అర్జీదారుని చంపుతానని బెదిరిస్తున్నాడు.",
    "అర్జీదారు ఒక పేద రైతు, అతనికి వేరే జీవనాధారం లేదు.",
    "అర్జీదారు ప్రతివాదికి {amount} రూపాయలు అప్పుగా ఇచ్చాడు.",
    "ఆ డబ్బు ఇప్పటివరకు తిరిగి ఇవ్వలేదు.",
    "ఎన్నిసార్లు అడిగినా ప్రతివాది కాలయాపన చేస్తున్నాడు.",
    "గ్రామంలోని చాలా మంది ఈ సంఘటనకు సాక్షులుగా ఉన్నారు.",
    "అర్జీదారుని తండ్రి తేదీ {date} న మరణించారు.",
    "వారి ఆస్తి పంపకం ఇంతవరకు జరగలేదు.",
    "ప్రతివాది తప్పుడు పత్రాలు తయారు చేయించాడు.",
    "అర్జీదారు అనేకసార్లు అధికారులకు ఫిర్యాదు చేసినా ఎవరూ పట్టించుకోలేదు.",
    "తేదీ {date} రాత్రి సుమారు {hour} గంటలకు ఇంటి తాళం పగలగొట్టి దొంగతనం చేశారు.",
    "ఆ రోజు అర్జీదారు కుటుంబంతో కలిసి వేరే ఊరికి వెళ్ళాడు.",
    "తిరిగి వచ్చి చూడగా ఇంట్లో వస్తువులు కనిపించలేదు.",
    "ఈ పని ఎవరో తెలియని వ్యక్తులు చేసినట్లు అనుమానం ఉన్నది.",
    "ప్రతివాది అర్జీదారుని నానా మాటలు అని కొట్టినాడు.",
    "ఈ విషయం గ్రామ పెద్దలకు కూడా తెలుసు.",
    "ప్రతివాది పొలంలోకి అక్రమంగా ప్రవేశించి పంటను నాశనం చేశాడు.",
    "ఈ విషయమై గ్రామ పంచాయతీలో కూడా ఫిర్యాదు చేయడం జరిగింది.",
    "పంచాయతీ పెద్దలు చెప్పినా ప్రతివాది వినిపించుకోలేదు.",
    "అర్జీదారు వద్ద ఆ భూమికి సంబంధించిన అన్ని పత్రాలు ఉన్నాయి.",
    "పట్టాదారు పాసు పుస్తకం అర్జీదారుని పేరు మీదనే ఉన్నది.",
    "ప్రతివాది రాత్రి సమయంలో వచ్చి గొడవ చేసి వెళ్ళాడు.",
    "ఆ సమయంలో అర్జీదారుని భార్య మరియు పిల్లలు ఇంట్లోనే ఉన్నారు.",
    "ఈ సంఘటన వలన అర్జీదారు కుటుంబం భయంతో జీవిస్తున్నది.",
    "అప్పు తీసుకున్నప్పుడు ప్రతివాది ప్రామిసరీ నోటు రాసి ఇచ్చాడు.",
    "ఆ నోటు అసలు ప్రతి అర్జీదారు వద్ద భద్రంగా ఉన్నది.",
    "ప్రతివాది ఆరు నెలలలో డబ్బు తిరిగి ఇస్తానని మాట ఇచ్చాడు.",
    "అర్జీదారు పలుమార్లు ఇంటికి వెళ్ళి అడిగినా ఫలితం లేదు.",
    "అర్జీదారుకు వయస్సు పైబడినందున పని చేయలేని స్థితిలో ఉన్నాడు.",
    "ఈ కేసులో సాక్ష్యం చెప్పడానికి గ్రామస్తులు సిద్ధంగా ఉన్నారు.",
    "ఇంట్లో దాచిన బంగారు గొలుసు మరియు నగదు కనిపించలేదు.",
    "అర్జీదారు ఈ విషయాన్ని వెంటనే పోలీసు స్టేషనుకు తెలియజేశాడు.",
]
CLAUSES = [
    "ఈ విషయమై అర్జీదారు మరే ఇతర న్యాయస్థానములో దావా వేయలేదు.",
    "పైన తెలిపిన విషయాలన్నీ అర్జీదారుకు తెలిసినంతవరకు నిజమైనవి.",
    "అర్జీదారు న్యాయస్థానం యొక్క ప్రతి ఆదేశాన్ని పాటిస్తాడు.",
    "అర్జీదారు బెయిల్ షరతులన్నింటినీ పాటిస్తాడు.",
]
PRAYERS = ["కావున అయ్యా గారు తగిన ఉత్తర్వులు జారీ చేయవలసిందిగా ప్రార్థన.",
           "కావున ప్రతివాదిపై తగిన చర్య తీసుకోవలసిందిగా మనవి.",
           "కావున నా ఫిర్యాదును స్వీకరించి న్యాయం చేయవలసిందిగా కోరుచున్నాను."]
THANKS = "అర్జీదారు ఎల్లప్పుడూ కృతజ్ఞుడై ఉంటాడు."
VERIFY = "పైన తెలిపిన విషయాలన్నీ నాకు తెలిసినంతవరకు సత్యమని ప్రమాణపూర్వకంగా తెలియజేయుచున్నాను."
MARGIN_NOTES = ["నకలు అందినది", "హాజరు కావలెను", "దాఖలు చేయబడినది", "విచారణ తేదీ {date}"]


def _date(rng) -> str:
    return f"{int(rng.integers(1, 29)):02d}/{int(rng.integers(1, 13)):02d}/{int(rng.integers(2015, 2026))}"


def sample_document(rng: np.random.Generator, ok: Callable[[str], bool] = lambda s: True,
                    length: float = 1.0) -> dict:
    """Blocks of one document, in reading order, in the format of
    legal_text.sample_document. `length` scales the number of facts (small
    writing needs more text to fill a page)."""
    def pick(options):
        good = [o for o in options if ok(o)]
        return str(rng.choice(good or options))

    family = pick(FAMILY)
    name = f"{family} {pick(GIVEN_M)}"
    father = f"{family} {pick(GIVEN_M)}"
    district = pick(DISTRICTS)
    village = pick(VILLAGES)
    address = (f"ఇంటి నంబరు {int(rng.integers(1, 400))} {village} {district} జిల్లా" if rng.random() < 0.5
               else f"{village} గ్రామం {district} జిల్లా")
    slots = {"district": district, "name": name, "father": father, "address": address,
             "age": str(int(rng.integers(22, 70))), "date": _date(rng), "num": str(int(rng.integers(11, 999))),
             "village": village, "years": str(int(rng.integers(5, 40))), "opp": f"{pick(FAMILY)} {pick(GIVEN_M)}",
             "amount": str(int(rng.integers(2, 90)) * 1000), "hour": str(int(rng.integers(1, 12)))}
    fill = lambda s: s.format(**slots)  # noqa: E731

    title = fill(pick(TITLES))
    affidavit = title == "ప్రమాణ పత్రము"
    blocks = [{"name": "title", "text": title, "align": "center", "underline": rng.random() < 0.6}]
    if not affidavit:
        blocks.append({"name": "subject", "text": "విషయం: " + pick(SUBJECTS), "align": "left"})
        blocks.append({"name": "salutation", "text": "అయ్యా,", "align": "left"})
    facts = [fill(f) for f in FACTS]
    facts = [f for f in facts if ok(f)] or facts
    facts = list(rng.choice(facts, size=min(len(facts), int(rng.integers(9, 15) * length)), replace=False))
    n = len(facts)  # more than fits; the page layout drops sentences to fill one sheet
    paras = [[fill(OATH if affidavit else OPENING)] + facts[: n // 3], facts[n // 3 : 2 * n // 3], facts[2 * n // 3 :]]
    if affidavit or rng.random() < 0.5:
        paras.append(list(rng.choice(CLAUSES, size=int(rng.integers(2, 4)), replace=False)))
    paras.append([VERIFY] if affidavit else [pick(PRAYERS), THANKS])
    for i, p in enumerate(q for q in paras if q):
        blocks.append({"name": f"paragraph_{i + 1}", "sentences": [str(s) for s in p], "align": "left", "indent": True})
    blocks.append({"name": "signer", "text": "ప్రమాణకర్త" if affidavit else "అర్జీదారు", "align": "right"})
    blocks.append({"name": "signature", "text": name, "align": "right", "signature": True})
    blocks.append({"name": "date", "text": f"తేదీ: {slots['date']}", "align": "left"})
    blocks.append({"name": "place", "text": f"స్థలం: {district}", "align": "left"})
    return {"blocks": blocks, "margin_note": fill(pick(MARGIN_NOTES)), "slots": slots}
