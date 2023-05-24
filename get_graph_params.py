import lib as l

lang_combos = [["ger", "fr"], ["ger", "sp"], ["ger", "eng"], ["eng", "fr"], ["eng", "sp"], ["sp", "fr"]]
keywords = ["telephone", "iron", "gasoline"]

dump = open("dump.txt")

for kw in keywords:
    for combo in lang_combos:
        embeddings = l.EMB.Embeddings("gasoline", ["sp", "eng", "ger"])