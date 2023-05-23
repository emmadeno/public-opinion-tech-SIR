import lib as l

def get_year(path):
        return path.split("/")[-1].split("_")[1]

def lemmas_to_list(row):
    if "c'est-à-dire" in row: row = row.replace("c'est-à-dire", "c est à dire")
    return l.json.loads(row.replace("'", '"'))

# return dict [indexes] -> distr
def flatten_topic_assignments(topics, distribution):
    topics_set = set([i for i in topics if i != "random"])
    final_dict = {}
    for topic in topics_set:
        indexes = [i for i, x in enumerate(topics) if (x == topic and x != "random")]
        total_distr = 0
        for i in indexes: total_distr += distribution[i]
        final_dict[tuple(indexes)] = total_distr
    return final_dict

def get_corpus(lemmas):
    corpus = l.tp.utils.Corpus()
    for doc in lemmas:
        if doc: corpus.add_doc(doc)
    return corpus

def get_topic_descriptors(model, index, top_n=15):
    return str([item[0] for item in model.get_topic_words(index, top_n=top_n)])

class Timespan:
    def __init__(self, df, kw, lang, index):
        self.df_total = df
        self.kw = kw
        self.lang = lang
        self.index = index

        self.df = None
        self.model = None
        self.corpus = None
        self.year = None
        self.topics = None

        self.get_df_and_model()
        self.print_topic_descriptors()
        

    def get_df_and_model(self):
        paths = l.glob.glob("../../data/spacy_models/" + self.kw + "/models_" + self.lang + "/*")
        paths.sort()
        self.year = get_year(paths[self.index])
        next_year = get_year(paths[self.index + 1]) if self.index < len(paths) - 1 else "2023"

        self.df = self.df_total[(self.df_total['date']>= self.year + '-01-01') & (self.df_total['date'] < next_year + '-01-01')]
        self.df.lemmas = self.df.lemmas.apply(lambda x : lemmas_to_list(x))

        print("Timespan from " + self.df .date.min().strftime(l.C.DATE_FORMAT) + " to " 
              + self.df .date.max().strftime(l.C.DATE_FORMAT) + " has " + str(len(self.df )) + " articles.")
        
        self.model = l.tp.PAModel.load(paths[self.index])
        self.corpus = get_corpus(self.df.lemmas.values)
        print("Successfully loaded model.")
    
    def train_model(self, k1=1, k2=5, save=True, name=None):
        self.model = l.tp.PAModel(tw=l.tp.TermWeight.IDF, min_df=10, k1=k1, k2=k2, corpus=self.corpus, seed=0)
        self.model.burn_in=100
        self.model.train(1000, workers=1)

        if save and name is not None:
            self.model.save(name, full=False)
        elif save:
             name = "../../data/spacy_models/" + self.kw + "/models_" + self.lang + "/model_" + self.year + "_" + self.lang + "_" + self.kw + ".obj"
             self.model.save(name, full=False)

    def print_topic_descriptors(self, top_n=15):
        for k in range(self.model.k2) : print("TOPIC " + str(k) + "\n" + get_topic_descriptors(self.model, k, top_n))

    def get_topic_of_doc(self, row) :
        if not row.lemmas:
            return row
        infered, _ = self.model.infer(self.model.make_doc(row.lemmas))
        sorted_indices = l.np.argsort(infered[1])[::-1]
        curr_ind = 0
        
        while "random" in self.topics[sorted_indices[curr_ind]]:
            curr_ind += 1

        row["topic_id"] = sorted_indices[curr_ind]
        row["topic"] = self.topics[sorted_indices[curr_ind]]
        return row

    def assign_topics(self, topics, thresh=0.00001):
        self.topics = topics
        self.df["topic_id"] = -1
        self.df["topic"] = ""
        self.df = self.df.apply(lambda x : self.get_topic_of_doc(x), axis = 1)
        self.df = self.df[self.df.topic_id != -1]
        self.get_keyword_topics(thresh)

    def get_keyword_topics(self, thresh):
        m = l.C.SPECIAL_CHARACTER_MAPS[self.lang]
        for kw in l.U.get_kw_dict(self.kw)[self.lang]:
            prob_dist = {}
            arr = ["".join(m.get(ch, ch) for ch in item.lower()) for item in self.model.used_vocabs]
            if kw in arr:
                for k in range(self.model.k2):
                    if self.topics[k] != "random":
                        prob_dist[k] = self.model.get_topic_word_dist(k)[arr.index(kw)]
                flattened_distr = dict(sorted(flatten_topic_assignments(self.topics, prob_dist).items(), key=lambda item: item[1] ,reverse=True))
                print("Keyword " + kw + " most likely belongs to topics :")
                for key, v in flattened_distr.items():
                    if v > thresh:
                        print("  - " + str(self.topics[key[0]]).upper() + " : topic(s) " + ", ".join([str(x) for x in key]))
            else:
                print("Keyword " + kw + " not in vocabulary used to train model.")

    def save(self):
        topics_df = l.pd.read_csv("../../data/topics.csv")
        count = self.df.groupby("topic_id").count()
        rows = []  
        for i, name in enumerate(self.topics):
            id = "_".join([self.lang, self.year, self.kw, str(i)])
            count_row = count.loc[i]["date"] if i in count.index.values else 0
            rows.append([id, self.lang, self.year, self.kw, get_topic_descriptors(self.model, i), name, count_row])
        to_concat = [topics_df, l.pd.DataFrame(rows, columns=topics_df.columns.values)]
        new_df = l.pd.concat(to_concat)
        new_df.to_csv("../../data/topics.csv", index=False)
                    