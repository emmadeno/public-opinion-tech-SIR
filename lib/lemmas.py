import lib as l
import lib.constants as C

# SCRIPT ADAPTED FROM (C) Germans Savcisens
# Url containing scripts : https://github.com/carlomarxdk/topic_modelling

def load_spacy(lang):
    return l.spacy.load(C.SPACY_LANGS[lang], disable=["ner",  "entity_linker",   "parser", 
                                           "textcat", "textcat_multilabel",  "senter",  "sentencizer",  "transformer"
                                          ])

def preproc(lang, text):
    sp = load_spacy(lang)
    all_lemmas = []
    for article in text:
        all_lemmas.append([article[0], [word.lemma_ for word in sp(article[1]) if ((word.is_alpha) and (not word.is_stop) and (len(word.lemma_) > 3))]])
    return all_lemmas

def chunker(iterable, chunksize):
    return (iterable[pos: pos + chunksize] for pos in range(0, len(iterable), chunksize))

def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]
    
def process_df(df, num_processors, lang):
    df.insert(0, 'id', range(0,len(df)))
    df_to_lemmatize = df.drop(columns = C.COLUMNS_TO_DROP[lang]).values.tolist()
    chunk_size = l.math.ceil((len(df_to_lemmatize)/num_processors)/10)*10
    chunks = chunker(df_to_lemmatize, chunk_size)
    with l.Pool(num_processors) as p:
       res = p.map(l.partial(preproc, lang), chunks)
    df_lemmas = l.pd.DataFrame(flatten(res), columns=["id", "lemmas"])
    df_complete = df.merge(df_lemmas, on="id")
    print(len(df_complete))
    return df_complete.drop(columns = ["id"])