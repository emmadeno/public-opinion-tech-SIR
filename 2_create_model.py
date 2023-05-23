# SCRIPT ADAPTED FROM (C) Germans Savcisens
# Url containing scripts : https://github.com/carlomarxdk/topic_modelling

import lib as l
import lib.utils as U
import lib.constants as C
import lib.keywords as KW
import lib.lemmas as SP
import lib.processing as PR



class TopicWrapper(l.BaseEstimator):
    """ 
    Wrapper for the Tomotopy HPA model. It simplifies the hyperparamaeter search with Sklearn.
    """
    def __init__(self,  
                k1: int, 
                k2: int, 
                top_n:int = 25, 
                train_iter: int = 500,
                random_state: int = 0,
                num_workers: int = 1,
                ) -> None:
        super().__init__()
        self.random_state = random_state
        self.k1 = k1
        self.k2 = k2
        self.train_iter = train_iter
        self.top_n = top_n
        self.num_workers = num_workers
        self.model = None

    def __init_model__(self):
        """Initialisez the HPA model with specific parameters"""
        return l.tp.PAModel(tw=l.tp.TermWeight.PMI, min_cf=10, rm_top=1, 
                          k1=self.k1, k2=self.k2, seed=self.random_state)
    def fit(self, X, **kwargs):
        corpus = l.tp.utils.Corpus()
        for doc in X:
            if doc: 
                corpus.add_doc(doc)
        self.model = self.__init_model__()
        self.model.add_corpus(corpus)
        self.model.burn_in = 100
        self.model.train(self.train_iter, workers=self.num_workers)
        return self

    def predict(self, X):
        infered_corpus, ll = self.model.infer(X)
        return infered_corpus, ll
    def score(self, *args, **kwargs) -> float:
        """Returns the coherence score"""
        return -l.tp.coherence.Coherence(self.model,coherence="u_mass").get_score()
    def set_params(self, **params):
        self.model = None
        return super().set_params(**params)
    
def get_best_params(df, lang, year, kw, params):
    lemmas = l.PR.get_lemmas(df)

    model = TopicWrapper(k1=1, k2=1, top_n=50, num_workers=1, train_iter=750) # initialise simple model 
    num_splits = 4 #cv_splits
    param_grid = list()

    # For HPA model: k1 <= k2
    # for other datasets you should probably increae the ranges 
    for i in range(1,3):
        for j in range(i, 15):
            param_grid.append({"k1": [i], "k2": [j]})

    search = l.GridSearchCV(model, param_grid, cv=num_splits, n_jobs=20, verbose=2)
    result = search.fit(lemmas)

    se = l.np.array([std/l.np.sqrt(num_splits) for std in result.cv_results_["std_test_score"]])
    means = result.cv_results_["mean_test_score"]
    best_id = result.best_index_
    cutoff = means[best_id] - se[best_id]

    optimal_id = l.np.argwhere(means>cutoff)[0]
    optimal_params = param_grid[optimal_id.item()]

    row = [lang, kw, year, str(result.best_params_["k1"]), str(result.best_params_["k2"]), optimal_params["k1"][0], optimal_params["k2"][0]]
    params = params.append(row)

    if optimal_params["k2"][0] < 3 and result.best_params_["k2"] >= 3:
         return (lemmas, result.best_params_["k1"], result.best_params_["k2"], params)

    return (lemmas, optimal_params["k1"][0], optimal_params["k2"][0], params)


def get_all_models(lang, keyword, begin_years, params):
    
    country = C.LANG_TO_COUNTRY[lang]
    print("Loading data ! -- " + country + " " + keyword)
    keyword_df = U.load_df(C.LANG_TO_COUNTRY[lang], keyword=keyword)

    for i, year in enumerate(begin_years):
        timespan_start = l.time.time()
        date_begin = str(year) + "-01-01"
        timespan = None
        if i == (len(begin_years)-1):
            timespan = keyword_df[(keyword_df['date']>=date_begin)]
        else:
            date_end = str(begin_years[i+1]) + "-01-01"
            timespan = keyword_df[(keyword_df['date']>=date_begin) & (keyword_df['date']<date_end)]
        (lemmas, k1, k2, params) = get_best_params(timespan, lang, year, keyword, params)

        PR.get_model(k1, k2, lemmas, U.get_model_name(lang, keyword, year))
        print("Model " + str(i + 1) + " for " + country + " done, took " + U.get_time(timespan_start))
        

    print(country + " done !")
    return params


def load_all_models(kw, years_dict):
    for lang in years_dict.keys():
        years = years_dict[lang]
        params = l.pd.read_csv("params.csv")
        params = get_all_models(lang, kw, years, params)
        params.to_csv("params.csv", index=False)


BEGIN_YEARS_TELEPHONE = {
    "fr" : [1870, 1890, 1900, 1910, 1920, 1925, 1930, 1935, 1940],
    "sp" : [1870, 1890, 1900, 1910, 1920, 1930],
    "ger" : [1880, 1920, 1935],
    "eng" : [1840, 1870, 1900]
}

BEGIN_YEARS_IRON = {
    "fr" : [1820, 1850, 1870, 1880, 1890, 1900, 1910, 1920, 1930],
    "sp" : [1860, 1880, 1890, 1900, 1910, 1920],
    "eng" : [1830, 1850, 1855, 1860, 1865, 1870, 1875, 1900],
    "ger" : [1880, 1900, 1910, 1920, 1930, 1940]
}


BEGIN_YEARS_GASOLINE = {
    "fr" : [1820, 1850, 1870, 1880, 1890, 1900, 1910, 1920, 1930],
    "sp" : [1860, 1880, 1890, 1900, 1910, 1920],
    "eng" : [1850, 1866, 1873, 1900],
    "ger" : [1880, 1900, 1910, 1920, 1930, 1940]
}

#load_all_models("gasoline", BEGIN_YEARS_GASOLINE)
load_all_models("iron", BEGIN_YEARS_IRON)
#load_all_models("telephone", BEGIN_YEARS_TELEPHONE)




