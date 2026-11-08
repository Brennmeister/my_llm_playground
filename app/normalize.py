"""Deterministic, locale-aware normalization of the raw strings the LLM extracted."""
from __future__ import annotations

import datetime as dt
import re
import unicodedata
from typing import Optional

# ---------------------------------------------------------------------------
# Language detection (keyword based, no extra dependency)
# ---------------------------------------------------------------------------

_LANG_KEYWORDS = {
    "de": ["bestellung", "bestellnummer", "menge", "lieferung", "liefertermin", "einzelpreis", "mwst", "gesamt", "artikel", "stück", "lieferant", "datum", "zahlungsbedingungen"],
    "en": ["purchase order", "order no", "quantity", "qty", "unit price", "delivery", "total", "amount", "supplier", "vendor", "item", "description", "payment terms"],
    "pl": ["zamówienie", "zamówienia", "ilość", "cena", "dostawa", "dostawy", "wartość", "netto", "brutto", "szt", "dostawca", "nabywca", "termin"],
    "it": ["ordine", "quantità", "prezzo", "consegna", "fornitore", "totale", "imponibile", "iva", "articolo", "descrizione", "pagamento", "pezzi"],
}


def detect_language(text: str) -> Optional[str]:
    low = text.lower()
    scores = {lang: sum(low.count(k) for k in kws) for lang, kws in _LANG_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"-?\d[\d.,'   ]*")


def parse_number(value: Optional[str], lang: Optional[str] = None) -> Optional[float]:
    """Parse '1.234,56', '1,234.56', '1 234,56 zł', "1'234.50", '12,5 %', '-3' ..."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    negative = s.startswith("-") or s.endswith("-") or (s.startswith("(") and s.endswith(")"))
    m = _NUM_RE.search(s.replace("−", "-"))
    if not m:
        return None
    num = re.sub(r"[\s'  ]", "", m.group(0)).lstrip("-").rstrip(".,")
    if not num:
        return None

    if "," in num and "." in num:
        dec = "," if num.rfind(",") > num.rfind(".") else "."
    elif "," in num:
        parts = num.split(",")
        # "1,234" / "1,234,567" in English = thousands; otherwise comma is decimal
        if lang == "en" and all(len(p) == 3 for p in parts[1:]):
            dec = None
        elif len(parts) > 2:
            dec = None
        else:
            dec = ","
    elif "." in num:
        parts = num.split(".")
        # "1.234" / "1.234.567" in de/pl/it = thousands
        if lang != "en" and all(len(p) == 3 for p in parts[1:]) and (len(parts) > 2 or lang in ("de", "pl", "it")):
            dec = None
        elif len(parts) > 2:
            dec = None
        else:
            dec = "."
    else:
        dec = None

    if dec is None:
        num = num.replace(",", "").replace(".", "")
    else:
        thousands = "." if dec == "," else ","
        num = num.replace(thousands, "").replace(dec, ".")
    try:
        f = float(num)
    except ValueError:
        return None
    return -f if negative else f


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

_MONTHS = {
    # de
    "januar": 1, "jänner": 1, "februar": 2, "märz": 3, "maerz": 3, "april": 4, "mai": 5, "juni": 6, "juli": 7,
    "august": 8, "september": 9, "oktober": 10, "november": 11, "dezember": 12,
    # en
    "january": 1, "february": 2, "march": 3, "may": 5, "june": 6, "july": 7, "october": 10, "december": 12,
    # pl (nominative + genitive)
    "styczeń": 1, "stycznia": 1, "luty": 2, "lutego": 2, "marzec": 3, "marca": 3, "kwiecień": 4, "kwietnia": 4,
    "maj": 5, "maja": 5, "czerwiec": 6, "czerwca": 6, "lipiec": 7, "lipca": 7, "sierpień": 8, "sierpnia": 8,
    "wrzesień": 9, "września": 9, "październik": 10, "października": 10, "listopad": 11, "listopada": 11,
    "grudzień": 12, "grudnia": 12,
    # it
    "gennaio": 1, "febbraio": 2, "marzo": 3, "aprile": 4, "maggio": 5, "giugno": 6, "luglio": 7, "agosto": 8,
    "settembre": 9, "ottobre": 10, "novembre": 11, "dicembre": 12,
}
_MONTH_ABBR = {
    "jan": 1, "feb": 2, "mar": 3, "mär": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
    "okt": 10, "oct": 10, "nov": 11, "dez": 12, "dec": 12, "gen": 1, "mag": 5, "giu": 6, "lug": 7, "ago": 8,
    "set": 9, "ott": 10, "dic": 12, "sty": 1, "lut": 2, "kwi": 4, "cze": 6, "lip": 7, "sie": 8, "wrz": 9,
    "paź": 10, "lis": 11, "gru": 12,
}


def _mk(y: int, m: int, d: int) -> Optional[str]:
    if y < 100:
        y += 2000
    try:
        return dt.date(y, m, d).isoformat()
    except ValueError:
        return None


def parse_date(value: Optional[str], lang: Optional[str] = None) -> Optional[str]:
    """Return ISO date (YYYY-MM-DD) or None. Handles numeric dates, month names in
    de/en/pl/it and calendar weeks ('KW 45/2026', 'CW45 2026', 'tydzień 45', 'settimana 45')."""
    if not value:
        return None
    s = str(value).strip().lower()

    # ISO 2026-11-08
    m = re.search(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b", s)
    if m:
        return _mk(int(m[1]), int(m[2]), int(m[3]))

    # Calendar week -> Monday of that ISO week
    m = re.search(r"\b(?:kw|cw|wk|week|woche|kalenderwoche|tydz(?:ień)?|tyg|settimana|sett)\.?\s*(\d{1,2})(?:\s*[/.\-]?\s*(\d{2,4}))?", s)
    if m:
        week = int(m[1])
        year = int(m[2]) if m[2] else dt.date.today().year
        year = year + 2000 if year < 100 else year
        try:
            return dt.date.fromisocalendar(year, week, 1).isoformat()
        except ValueError:
            return None

    # Numeric d.m.y / d/m/y / d-m-y (US m/d/y only if day>12 makes it unambiguous)
    m = re.search(r"\b(\d{1,2})[./\-](\d{1,2})[./\-](\d{2,4})\b", s)
    if m:
        a, b, y = int(m[1]), int(m[2]), int(m[3])
        if lang == "en" and "/" in m[0] and b > 12 >= a:
            return _mk(y, a, b)
        return _mk(y, b, a)

    # "8. November 2026", "8 listopada 2026", "8 novembre 2026", "November 8, 2026", "8 Nov 2026"
    words = {**_MONTHS, **_MONTH_ABBR}
    m = re.search(r"\b(\d{1,2})\.?\s+([^\W\d_]+)\.?,?\s+(\d{2,4})\b", s)
    if m and m[2] in words:
        return _mk(int(m[3]), words[m[2]], int(m[1]))
    m = re.search(r"\b([^\W\d_]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", s)
    if m and m[1] in words:
        return _mk(int(m[3]), words[m[1]], int(m[2]))
    return None


# ---------------------------------------------------------------------------
# Currency, country, units
# ---------------------------------------------------------------------------

_CURRENCIES = {"€": "EUR", "eur": "EUR", "euro": "EUR", "zł": "PLN", "zl": "PLN", "pln": "PLN", "$": "USD", "usd": "USD",
               "£": "GBP", "gbp": "GBP", "chf": "CHF", "czk": "CZK", "kč": "CZK", "huf": "HUF", "sek": "SEK", "dkk": "DKK"}


def normalize_currency(value: Optional[str], *fallback_texts: Optional[str]) -> Optional[str]:
    for candidate in (value, *fallback_texts):
        if not candidate:
            continue
        low = candidate.lower()
        if re.fullmatch(r"[a-z]{3}", low.strip()) and low.strip().upper() in _CURRENCIES.values():
            return low.strip().upper()
        for sym, code in _CURRENCIES.items():
            if (sym.isalpha() and re.search(rf"\b{re.escape(sym)}\b", low)) or (not sym.isalpha() and sym in low):
                return code
    return None


_COUNTRIES = {
    "DE": ["deutschland", "germany", "niemcy", "germania", "brd", "d"],
    "PL": ["polska", "poland", "polen", "polonia", "pl"],
    "IT": ["italia", "italy", "italien", "włochy", "wlochy", "i"],
    "AT": ["österreich", "oesterreich", "austria", "a"],
    "CH": ["schweiz", "switzerland", "szwajcaria", "svizzera", "suisse", "ch"],
    "FR": ["frankreich", "france", "francja", "francia", "f"],
    "NL": ["niederlande", "netherlands", "holandia", "paesi bassi", "nl"],
    "BE": ["belgien", "belgium", "belgia", "belgio", "b"],
    "CZ": ["tschechien", "czech republic", "czechia", "czechy", "repubblica ceca", "cz"],
    "ES": ["spanien", "spain", "hiszpania", "spagna", "españa", "e"],
    "GB": ["großbritannien", "united kingdom", "uk", "great britain", "england", "wielka brytania", "regno unito"],
    "US": ["usa", "united states", "vereinigte staaten", "stany zjednoczone", "stati uniti"],
    "DK": ["dänemark", "denmark", "dania", "danimarca", "dk"],
    "SE": ["schweden", "sweden", "szwecja", "svezia", "s"],
    "HU": ["ungarn", "hungary", "węgry", "ungheria", "h"],
    "SK": ["slowakei", "slovakia", "słowacja", "slovacchia", "sk"],
    "SI": ["slowenien", "slovenia", "słowenia", "slo"],
    "LU": ["luxemburg", "luxembourg", "luksemburg", "lussemburgo", "l"],
}
_COUNTRY_LOOKUP = {name: code for code, names in _COUNTRIES.items() for name in names}


def normalize_country(value: Optional[str], vat_id: Optional[str] = None, postal_code: Optional[str] = None) -> Optional[str]:
    if value:
        v = value.strip().lower().strip(".")
        if v in _COUNTRY_LOOKUP:
            return _COUNTRY_LOOKUP[v]
        if re.fullmatch(r"[a-z]{2}", v):
            return v.upper()
    if vat_id:
        m = re.match(r"\s*([A-Za-z]{2})", vat_id)
        if m and m[1].upper() in _COUNTRIES:
            return m[1].upper()
    if postal_code:  # "D-12345", "PL 00-950", "I-20121"
        m = re.match(r"\s*([A-Za-z]{1,3})[\s-]", postal_code)
        if m and m[1].lower() in _COUNTRY_LOOKUP:
            return _COUNTRY_LOOKUP[m[1].lower()]
        if re.fullmatch(r"\d{2}-\d{3}", postal_code.strip()):
            return "PL"
    return None


# UN/ECE Recommendation 20 unit codes
_UNITS = {
    "H87": ["stk", "stück", "st", "stck", "pcs", "pc", "piece", "pieces", "ea", "each", "szt", "sztuk", "sztuka", "pz", "pezzi", "pezzo", "nr", "n", "unit", "units", "x"],
    "C62": ["einheit", "jednostka", "unità"],
    "KGM": ["kg", "kilogramm", "kilogram", "kilogrammo", "kilo"],
    "GRM": ["g", "gramm", "gram", "grammo", "gr"],
    "TNE": ["t", "tonne", "ton", "tona", "tonnellata"],
    "MTR": ["m", "meter", "metre", "metr", "metro", "mb", "lfm", "lfdm", "ml"],
    "CMT": ["cm"],
    "MMT": ["mm"],
    "MTK": ["m2", "m²", "qm", "sqm", "mq"],
    "MTQ": ["m3", "m³", "cbm", "mc"],
    "LTR": ["l", "liter", "litre", "litr", "litro", "lt"],
    "SET": ["set", "sets", "satz", "sätze", "zestaw", "kpl", "komplet", "kit"],
    "PR": ["paar", "pair", "pairs", "para", "pary", "paio", "paia"],
    "XBX": ["karton", "box", "boxes", "ktn", "krt", "carton", "karton", "opak", "scatola", "scatole", "ctn"],
    "XPK": ["pack", "packung", "pkg", "pck", "pak", "paczka", "confezione", "conf", "vpe"],
    "XRO": ["rolle", "roll", "rolka", "rotolo", "rol"],
    "XPX": ["palette", "pallet", "paleta", "pal", "bancale"],
    "HUR": ["h", "std", "stunde", "stunden", "hour", "hours", "godz", "ora", "ore"],
}
_UNIT_LOOKUP = {name: code for code, names in _UNITS.items() for name in names}


def normalize_unit(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    v = unicodedata.normalize("NFC", value).strip().lower().rstrip(".")
    if v.upper() in _UNITS:
        return v.upper()
    return _UNIT_LOOKUP.get(v) or _UNIT_LOOKUP.get(re.sub(r"[^\w²³]", "", v))


def clean_str(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = re.sub(r"\s+", " ", str(value)).strip(" ,;:")
    return v if v and v.lower() not in ("null", "none", "n/a", "-", "--") else None


# ---------------------------------------------------------------------------
# Grounding: small models hallucinate - check that values really occur in the PDF text
# ---------------------------------------------------------------------------


def _squash(s: str) -> str:
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", s).lower())


class Grounder:
    """Checks whether an extracted value is present in the source text."""

    def __init__(self, text: str):
        self.squashed = _squash(text)
        self.digits = re.sub(r"\D", "", text)

    def has(self, value: Optional[str]) -> bool:
        if not value:
            return False
        sq = _squash(value)
        return bool(sq) and sq in self.squashed

    def has_digits(self, value: Optional[str]) -> bool:
        d = re.sub(r"\D", "", value or "")
        return len(d) >= 5 and d in self.digits


_EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$")


def valid_email(value: Optional[str]) -> Optional[str]:
    v = clean_str(value)
    return v if v and _EMAIL_RE.match(v) else None


def split_postal_city(postal: Optional[str], city: Optional[str], country: Optional[str]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Repair typical small-model mistakes: postal_code='80331 München', city='Deutschland'."""
    if city and city.strip().lower().strip(".") in _COUNTRY_LOOKUP:
        country = country or city
        city = None
    if postal and not city:
        tokens = postal.split()
        code = [t for t in tokens if re.search(r"\d", t)]
        name = [t for t in tokens if not re.search(r"\d", t)]
        if code and name:
            # keep country prefixes like "D-80331" / "PL 60-123" with the code
            postal, city = " ".join(code), " ".join(name)
    elif postal and city and city in postal:
        postal = postal.replace(city, "").strip(" ,")
    return postal or None, city or None, country
