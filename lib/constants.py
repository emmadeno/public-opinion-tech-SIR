COUNTRIES = ["Hamburg", "USA", "Spain", "France"]

DATE_FORMAT = "%Y"

SPECIAL_CHARACTER_MAPS = {
    "fr" : {"é" : "e", "à" : "a", "è" : "e", "ù" : "u", "â" : "a", "ê" : "e", "î" : "i", "ô" : "o", "û" : "u", "ç" : "c", "ë" : "e", "ï" : "i", "ü":"u", "œ":"oe"},
    "sp" : {"á" : "a", "é" : "e", "í" : "i", "ñ" : "n", "ó" : "o", "ú" : "u", "ü" : "u"},
    "ger" : {"ä" : "ae", "ö" : "oe", "ü" : "ue", "ß" : "ss"},
    "eng" : {}
}


SPACY_LANGS = {"eng" : "en_core_web_sm",
                "fr" : "fr_core_news_sm",
                "ger" : "de_core_news_sm",
                "sp" : "es_core_news_sm"}

COUNTRY_TO_LANG = {"USA" : "eng",
                    "France" : "fr",
                    "Hamburg" : "ger",
                    "Spain" : "sp"}

LANG_TO_COUNTRY = {"eng" : "USA",
                    "fr" : "France",
                    "ger" : "Hamburg",
                    "sp" : "Spain"}

DF_NAMES = {"eng" : "df_nyh",
                "fr" : "df_figaro",
                "ger" : "df_nhz",
                "sp" : "df_imparcial"}

RAW_DATA_PATH = "/LHSTdata/lhstdata1/denove/corpus-001.tar"

DATA_PATTERNS = {"USA" : "/*/**/*.xml",
                 "Hamburg" :"/*/**/*.xml",
                 "Spain" : "/*.txt",
                 "France" : "/*/**/*.json"
}

GASOLINE = {
    "eng" : ["gasoline", "petrol"],
    "fr" : ["petrole", "essence"],
    "ger" : ["benzin", "kraftstoff"],
    "sp" : ["gasolina", "bencina", "petroleo"]
}

ELECTRIC_LIGHT = {
    "eng" : ["electric", "light"],
    "fr" : ["lumiere", "electrique"],
    "ger" : ["elektrisch", "licht"],
    "sp" : ["luz", "electrica"]
}

TELEPHONE = {
    "eng" : ["telephone"],
    "fr" : ["telephone"],
    "ger" : ["telefon"],
    "sp" : ["telefono"]
}

IRON = {
    "eng" : ["iron"],
    "fr" : ["fer"],
    "ger" : ["eisen"],
    "sp" : ["hierro"]
}

ELECTRIC = {
    "eng" : ["electric"],
    "fr" : ["electrique"],
    "ger" : ["elektrisch"],
    "sp" : ["electric"]
}

COLUMNS_TO_DROP = {"fr":["date", "text"],
                    "eng":["date","text","height"],
                    "ger":["date", "height"],
                    "sp": ["date", "text"]} 