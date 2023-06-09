import lib as l

class Embeddings:

    def __init__(self, keyword, languages):
        self.languages = languages
        self.keyword = keyword

        self.embeddings = {}
        self.id2word = {}
        self.word2id = {}

        for i, lang in enumerate(self.languages):
            path = "./data/vectors/" + keyword + "/vectors-" + self.languages[i] + "-" + self.languages[(i + 1) % 2] + ".txt"
            if path != "./data/vectors/telephone/vectors-eng-ger.txt" and path != "./data/vectors/telephone/vectors-ger-eng.txt":
                embedding, id2word, word2id = load_vec(path)
                self.embeddings[lang] = embedding
                self.id2word[lang] = id2word
                self.word2id[lang] = word2id

        self.topics_df = self.load_topic_df()

    def load_topic_df(self):
        topics_df = l.pd.read_csv("data/topics.csv")
        topics_df = topics_df[topics_df.keyword == self.keyword]
        topics_df = topics_df.drop_duplicates()
        topics_df["topic_id"] = topics_df.apply(lambda row: "_".join([row.id, row.topic_name]).replace(" ", "_"), axis=1)
        return topics_df


def create_model(country, kw):
    df = l.U.load_df(country, keyword=kw)
    df["lemmas"] = df["lemmas"].apply(lambda x : x.replace("'", "").replace("[", "").replace("]", "").replace(",","").lower())
    sample = df.sample(1000)
    if country == "France" :
        sample = df[df['lemmas'].map(len) > 5000].sample(1000)
    elif country == "Spain" :
        sample = df[df['lemmas'].map(len) > 1000].sample(1000)
    print(sample.lemmas.apply(lambda x: len(x)).mean())
    sample['lemmas'].replace('', l.np.nan, inplace=True)
    sample.dropna(subset=['lemmas'], inplace=True)
    sample[['lemmas']].to_csv('train.txt', 
                                          index = False, 
                                          sep = ' ',
                                          header = None, 
                                          quoting = l.csv.QUOTE_NONE, 
                                          quotechar = "", 
                                          escapechar = " ")
    ftt_model = l.fasttext.train_unsupervised("train.txt", model='skipgram', minCount=1, epoch=15)
    ftt_model.save_model(country + "_" + kw + ".bin")
    l.subprocess.run(["rm", "train.txt"])


def load_vec(emb_path, nmax=50000):
    vectors = []
    word2id = {}
    with l.io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = l.np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = l.np.vstack(vectors)
    return embeddings, id2word, word2id


def add_embedding(t, word2id, embd, reduced_emb, reduced_word2id, idx):
    for w in l.U.to_list(t[1]):
          if w in word2id.keys():
              reduced_emb = [embd[word2id[w]]] if not len(reduced_emb) else reduced_emb + [embd[word2id[w]]]
              reduced_word2id[w] = idx
              idx += 1
    return reduced_emb, reduced_word2id, idx

def create_one_nbow(topic, reduced_word2id):
    bow0 = []
    w0 = [] #np.zeros((len(reduced_emb)), dtype=np.float32)
    for i, w in enumerate(topic):
        try :
            bow0 = [reduced_word2id[w]] if not len(bow0) else bow0 + [reduced_word2id[w]]
            w0 += [1 - 0.1*i]
        except :
            pass
    return bow0, w0

def compute_graph_data(embedding, threshold, threshold_relax):
    # aggregate all embeddings together
    reduced_emb = []
    reduced_word2id = dict()
    idx = 0
    for i, lang in enumerate(embedding.languages):
        for t in embedding.topics_df[(embedding.topics_df["lang"]==lang)].iterrows():
            try:
                reduced_emb, reduced_word2id, idx = add_embedding(t, embedding.word2id[lang], embedding.embeddings[lang], reduced_emb, reduced_word2id, idx)
            except:
                continue   
    reduced_emb = l.np.array(reduced_emb, dtype=l.np.float32)


    # create nbow
    nbow = dict()
    for i, lang in enumerate(embedding.languages):
        for i, t in enumerate(embedding.topics_df[embedding.topics_df["lang"]==lang].iterrows()):
            bow, w = create_one_nbow(l.U.to_list(t[1]), reduced_word2id)
            nbow[t[1].topic_id] = (t[1].topic_id, bow, l.np.ones(len(w))/len(w))
    
        
    # https://github.com/src-d/wmd-relax
    reduced_emb_t = l.np.array(reduced_emb, dtype=l.np.float32)
    calc = l.WMD(reduced_emb_t, nbow, vocabulary_min=1, vocabulary_max=2000)

    # calculate weight between each topic 
    graph_topics_df = l.pd.DataFrame(columns = ["from", "to", "weigth"])
    for t in embedding.topics_df.iterrows():
        topic = t[1].topic_id
        nn = calc.nearest_neighbors(topic, k=100, early_stop=0.99)
        for n in nn:
            # same country
            if (t[1].topic_id.split("_")[0] == embedding.topics_df[embedding.topics_df["topic_id"]==n[0]].iloc[0].topic_id.split("_")[0]) and n[1] < threshold:
                graph_topics_df.loc[len(graph_topics_df.index)] = [topic,embedding.topics_df[embedding.topics_df["topic_id"]==n[0]].iloc[0].topic_id, threshold - n[1]]
            if not(t[1].topic_id.split("_")[0] == embedding.topics_df[embedding.topics_df["topic_id"]==n[0]].iloc[0].topic_id.split("_")[0]) and n[1] < threshold_relax:
                graph_topics_df.loc[len(graph_topics_df.index)] = [topic,embedding.topics_df[embedding.topics_df["topic_id"]==n[0]].iloc[0].topic_id, threshold_relax - n[1]]

    return graph_topics_df

def find_communities(G):
    communities_greedy = l.community.greedy_modularity_communities(G)
    cov_greedy, perf_greedy = l.community.partition_quality(G, communities_greedy)
    mod_greedy = l.community.modularity(G, communities_greedy)
    communities_louvain = l.community.louvain_communities(G)
    cov_louvain, perf_louvain = l.community.partition_quality(G, communities_louvain)
    mod_louvain = l.community.modularity(G, communities_louvain)
    value_dict = {"greedy" : (mod_greedy, perf_greedy, cov_greedy), "louvain" : (mod_louvain, perf_louvain, cov_louvain)}
    return value_dict
    
def get_best_thresholds_parallel(embedding, thresholds):

    communities = []

    for thresholds_pair in thresholds:
        t, tr = thresholds_pair
        graph_df = compute_graph_data(embedding, threshold=t, threshold_relax=tr)
        G = l.nx.from_pandas_edgelist(graph_df, source='from', target='to')

        for algo in ["greedy", "louvain"]:
            try: 
                value_dict = find_communities(G)
                communities.append({"keyword" : embedding.keyword, "threshold" : t, "threshold_relax" : tr, "algo" : algo, "modularity": value_dict[algo][0], "performance": value_dict[algo][1], "coverage": value_dict[algo][2], "nodes": G.number_of_nodes()})
            except:
                communities.append({"keyword" : embedding.keyword, "threshold" : t, "threshold_relax" : tr, "algo" : algo, "modularity": "Nan", "performance": "Nan", "coverage": "Nan", "nodes": G.number_of_nodes()})


    return communities

def get_best_thresholds(embedding, num_processors=40):

    print("Starting : " + embedding.keyword + " " + str(embedding.languages))

    list_thresholds = []
    for t in l.np.arange(0.5, 7, 0.25):
        for tr in l.np.arange(t+0.25, 7, 0.25):
            list_thresholds.append((t, tr))

    
    chunks = l.SP.chunker(list_thresholds, 20)
    start = l.time.time()
    with l.Pool(num_processors) as p:
        communities = p.map(l.partial(get_best_thresholds_parallel, embedding), chunks)

    print("Multiprocessing done, took " + l.U.get_time(start))
    communities_df = l.pd.DataFrame.from_records(l.SP.flatten(communities))

    path = "_".join([embedding.keyword, embedding.languages[0], embedding.languages[1]])
    communities_df.to_csv("./dump/communities/" + path + ".csv")
    communities_df["nodes_percentage"] = communities_df.apply(lambda row: row["nodes"]/len(embedding.topics_df), axis=1)

    reduced_comm_df = communities_df[communities_df["nodes_percentage"] > 0.45]
    reduced_comm_df[["modularity", "performance", "coverage"]] = reduced_comm_df[["modularity", "performance", "coverage"]].apply(l.pd.to_numeric)

    t, tr, algo = find_best_thresholds(reduced_comm_df)

    return t, tr, algo


def find_best_thresholds(reduced_comm_kw_df):
    if len(reduced_comm_kw_df) == 0:
        print("rip")
        return (0.5, 0.5, "louvain")
    reduced_comm_kw_df["score"] = reduced_comm_kw_df.apply(lambda row: 0.7*row["modularity"] + 0.15*row["performance"] +0.15*row["coverage"], axis=1)
    best_line = reduced_comm_kw_df.loc[reduced_comm_kw_df["score"].sort_values().tail(1).index[0]]

    t, tr, algo = best_line["threshold"], best_line["threshold_relax"], best_line["algo"]
    print("Best parameters : t = " + str(t) + " ; tr = " + str(tr) + " ; algo = " + algo)

    return t, tr, algo

def save_graph_communities(t, tr, embedding, path):
    graph_data = l.EMB.compute_graph_data(embedding, t, tr) 
    #path = "./data/graphs/graph_" + embedding.keyword + ".csv"
    if l.os.path.exists(path):
        to_concat = [l.pd.read_csv(path), graph_data]
        graph_data = l.pd.concat(to_concat)
    graph_data.to_csv(path, index=False)



    
