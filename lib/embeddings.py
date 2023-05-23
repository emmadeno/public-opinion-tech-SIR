import lib as l

def create_model(country, kw):
    df = l.U.load_df(country, keyword=kw)
    df["lemmas"] = df["lemmas"].apply(lambda x : x.replace("'", "").replace("[", "").replace("]", "").replace(",","").lower())
    sample = df.sample(1000)
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
    ftt_model.save_model("kw_" + kw + "/ftt_models/" + country + "_" + kw + ".bin")
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
    
    
