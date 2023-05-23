import lib as l
import lib.processing as PR

def get_word_count(text, word):
    return str(text).lower().count(word)

def plot_occurences(df, keyword, words, logy=False):
    #PLOT NUMBER OF GASOLINE OCCURENCES IN ARTICLES 
    key = "num_of_" + keyword
    df[key] = 0
    for word in words:
        df[key] += df["article"].apply(lambda x : str(x).lower().count(word))
    group_by_num_of_keywords = df.groupby(key).count()["article"].reset_index()
    l.plt.bar(group_by_num_of_keywords[key], group_by_num_of_keywords["article"] )
    l.plt.title("Distribution of number of occurences of " + keyword + " by article")
    if logy:
        l.plt.yscale("log")
    l.plt.show()

def plot_articles_over_time(df, keyword):
    articles_by_year = df.groupby(df.date.dt.year)["article"].count().reset_index()
    l.plt.bar(articles_by_year.date, articles_by_year["article"])
    l.plt.title("Number of articles containing " + keyword + " by year")
    l.plt.show()

def plot_num_of_articles_per_topic(timespan):
    topic_data = timespan.df.groupby("topic").count()[["date"]].rename(columns={"date":"count"}).sort_values("count", ascending=False).reset_index()
    l.plt.bar(topic_data["topic"].values, topic_data["count"].values, width=0.3)
    l.plt.xticks(rotation=70)
    l.plt.title("Number of documents per topic")
    l.plt.ylabel("# of documents")
    l.plt.show()


def get_top_topics(topics, n=10):
    all_topics = l.pd.concat(topics)
    grouped = all_topics.groupby("TOPIC").sum().reset_index().sort_values("count", ascending=False).head(n)
    l.plt.bar(grouped["TOPIC"].values, grouped["count"].values, width=0.3)
    l.plt.xticks(rotation=70)
    l.plt.title("Number of documents per topic")
    l.plt.ylabel("# of documents")
    l.plt.show()

