import lib as l
import lib.constants as C

def to_list(list):
    return list.topics.replace("[", "").replace("]", "").replace("\'", "").split(", ")

def get_kw_dict(kw):
    if kw == "telephone": return l.C.TELEPHONE
    if kw == "gasoline": return l.C.GASOLINE
    if kw == "iron" : return l.C.IRON


def get_time(start):
    seconds = l.time.time() - start
    minutes = seconds / 60
    if minutes < 1:
        return str(round(seconds, 3)) + " seconds"
    else:
        return str(round(minutes, 3)) + " minutes"

def compress(df, name):
    compression_opts = dict(method='zip',
                        archive_name=name)
    df.to_csv(name, index=False,
            compression=compression_opts) 
    
def load_spacy(lang):
    return l.spacy.load(C.SPACY_LANGS[lang], disable=["ner",  "entity_linker",   "parser", 
                                           "textcat", "textcat_multilabel",  "senter",  "sentencizer",  "transformer"
                                          ])
    
def load_df(country, is_raw=False, keyword=None, is_parent=True):
    start = l.time.time()
    lang = C.COUNTRY_TO_LANG[country]
    df_path = "/scratch/students/denove/public-opinion-tech-SIR/data/newspaper_dfs/" + C.DF_NAMES[lang]  + ".zip"
    if keyword is not None:
        df_path = get_keyword_dataframe_name(lang, keyword, is_parent)
    df = l.pd.read_csv(df_path)
    df.date = l.pd.to_datetime(df.date, infer_datetime_format=True)
    if is_raw and lang == "fr":
        df.text = df['text'].apply(lambda x : str(x).replace("é", "e").replace("è","e").replace("ê","e").replace("à", "a").replace("ë", "e"))
    elif is_raw and lang == "sp":
        df.text = df['text'].apply(lambda x : str(x).replace("é","e").replace("á","a").replace("í", "i").replace("ó","o").replace("ú","u").replace("ü","u"))
    elif is_raw and lang == "ger":
        df.text = df['text'].apply(lambda x : str(x).replace("ä","ae").replace("ü","ue").replace("ö", "oe"))
    elif keyword is None and lang == "eng":
        df.height = df.height.apply(lambda x : l.json.loads(x))
    print("Loaded data from " + country + ", took " + get_time(start))
    print("Dataset has size " + str(df.shape) + " and columns " + str(df.columns.values))
    return df

def get_keyword_dataframe_name(lang, keyword, is_parent):
    df_name = C.DF_NAMES[lang]
    if is_parent:
        return "data/keyword_dfs/" + keyword + "/" + df_name + "_" + keyword + ".zip"
    else :
        return "../../data/keyword_dfs/" + keyword + "/" + df_name + "_" + keyword + ".zip"

def get_model_name(lang, keyword, begin_year):
    name = "model_" + str(begin_year) + "_" + lang + "_" + keyword + ".obj"
    return "data/spacy_models/" + keyword + "/models_" + lang + "/" + name
