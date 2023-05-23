import lib as l
import lib.utils as U
import lib.constants as C


# SCRIPT INSPIRED FROM (C) Elisa Michelet
# Url containing scripts : https://github.com/arobaselisa/industrial-west/tree/main

#######################
### SPLIT ARTICLES ###
#######################


def all_keywords_in_text(text, keywords):
    return all(keyword in str.lower(text) for keyword in keywords)

def any_keywords_in_text(text, keywords):
    return any(keyword in str.lower(text) for keyword in keywords)

def get_keyword_articles(articles, keywords, isAnd=False):
    filtered_txt = ""
    for article in articles:
        if isAnd:
            if all_keywords_in_text(article, keywords):
                filtered_txt += " " + article
        else :
            if any_keywords_in_text(article, keywords):
                filtered_txt += " " + article
    return filtered_txt

def ret_article_array(article, kw_articles, keywords, isAnd, lang=None):
    if isAnd and C.ELECTRIC[lang] in str.lower(article):
        kw_articles.append(article)
    elif not isAnd and any_keywords_in_text(article, keywords):
        kw_articles.append(article)
    return kw_articles

def get_newlines(x):
    newlines = []
    for i in range(len(x)):
        s = r'\n'
        if x[i] == s[0] and x[i + 1]==s[1]:
            newlines.append(i)
            i+=1
    return newlines

def num_cap(x):
    return sum(1 for c in x if c.isupper())

# returns articles in full text x
def split_articles_le_figaro(x, keywords, isAnd):
    articles = []
    regex = l.re.compile('[^a-zA-Z]')
    newlines = get_newlines(x)
    last_article_begin = 0
    for newline in newlines:
        if x[newline+2].isupper():
            to_check = regex.sub('', x[newline:newline+30])
            num_of_uppercase = sum(1 for c in to_check if c.isupper())
            if(num_of_uppercase > 2/3 * len(to_check)):
                articles.append(x[last_article_begin:newline])
                last_article_begin = newline

    articles.append(x[last_article_begin:])

    kw_articles = []
    for article in articles:
        article = article.replace(r'\n', "")
        if isAnd and C.ELECTRIC["fr"] in str.lower(article):
            kw_articles.append(article)
        elif not isAnd and any_keywords_in_text(article, keywords):
            kw_articles.append(article)
    
    return kw_articles


def split_articles_imparcial(row, keywords, isAnd=False):
    articles = row.split("\n")
    kw_articles = []
    for article in articles:
        if len(article) > 20:
            kw_articles = ret_article_array(article, kw_articles, keywords, isAnd, lang="sp")
            
    if(len(kw_articles) == 0):
        return [row]
    
    return kw_articles


# returns all articles in text "row" that include any of "keywords"
# "row" is a row of a Pandas Dataframe with columns "text" (whole text of file) and "height" (array of the "height" attribute in the XML)
def split_articles_nyh(row, keywords, isAnd=False):
    diff = l.np.diff(row.height)
    text_list = row.text.split(" ")
    titles = l.np.array(row.height)
    l_big = l.np.array([titles > 250], dtype=bool).astype(int).nonzero()[1]
    diff_2 = l.np.diff(l_big)
    if len(l_big) > 0:
        diff_2 = l.np.insert(diff_2, 0, 0)
    l_big_keep = l_big[diff_2 > 50]
    l_list = l.np.array([diff < -20], dtype=bool).astype(int).nonzero()[1]
    l_list = l.np.concatenate((l_list, l_big_keep))
    article_start = []
    for i in l_list:
        if num_cap(text_list[i]) > 2/3 * len(text_list[i]) and len(text_list[i]) > 1 and num_cap(text_list[i-1]) > 2/3 * len(text_list[i-1]):
            article_start.append(i)
    article_start.append(len(text_list) - 1)
    
    articles = ""
    for j in range(1, len(article_start)):
        index_start = article_start[j-1]
        index_end = article_start[j]
        last_article = " ".join(text_list[index_start:index_end])
        if isAnd:
            if C.ELECTRIC["eng"] in str.lower(last_article):
                articles+=(last_article)
        else:
            if any_keywords_in_text(last_article, keywords):
                articles+=(last_article)

    if len(articles) == 0:
        return row.text
    
    return articles


#######################
### DETECT KEYWORDS ###
#######################


def are_words_close(row, closeness, word1, word2):
    if word1 in row and word2 in row:
        indices_1 = [i for i, x in enumerate(row.split(" ")) if x == word1]
        indices_2 = [i for i, x in enumerate(row.split(" ")) if x == word2]
        permutations = [(x,y) for x in indices_1 for y in indices_2]
        for (x, y) in permutations:
            if l.np.abs(x - y) < closeness:
                return True
    return False

def areKeywordsInRow(row, keywords, isAnd=False):
    if str(row) == "nan":
        return False
    if isAnd:
        if all(text in str.lower(row) for text in keywords):
            #if are_words_close(row, 1000, keywords[0], keywords[1]):
                return True
    else:
        if any(text in str.lower(row) for text in keywords):
            return True
    return False   

def get_keyword_df(df, keywords, lang, isAnd=False):
    start = l.time.time()
    df_filtered = df.loc[df.text.apply(lambda x : areKeywordsInRow(x, keywords,isAnd))]
    print("Filtering for words " + str(keywords) + " done, took " + U.get_time(start))
    print("Size is " + str(df_filtered.shape) + ".")
    start = l.time.time()
    if lang == "fr":
        df_filtered["article"] = df_filtered.text.apply(lambda x : split_articles_le_figaro(x, keywords, isAnd))
    elif lang == "eng":
        df_filtered["article"] = df_filtered.apply(lambda x : split_articles_nyh(x, keywords, isAnd), axis=1)
    elif lang == "sp":
        df_filtered["article"] = df_filtered.text.apply(lambda x : split_articles_imparcial(x, keywords, isAnd))
    if lang != "ger":
        df_filtered = df_filtered.explode("article")
        print("Splitting articles done, took " + U.get_time(start) + ". New length is : " + str(df_filtered.shape))
    else:
        df_filtered = df_filtered.rename(columns={"text":"article"})
    return df_filtered