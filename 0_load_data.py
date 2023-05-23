# SCRIPT ADAOTED FROM (C) Elisa Michelet
# Url containing scripts : https://github.com/arobaselisa/industrial-west/tree/main

import lib as l
import lib.constants as C
import lib.utils as U


#### SCRIPT TO EXTRACT RELEVANT DATA FROM COMPRESSED FILES
# Form of input : python3 load_data [countries] -del -extr
# [countries] : list of countries to extract (i.e. "USA Hamburg Spain")
# flag -del : if added, deletes folders after extracting
# flag -extr : if added, extracts files

def get_country_files(country, file_type, all_files):
    country_name = "United States" if country == "USA" else country
    return [file_name for file_name in all_files.getnames() if country_name in file_name and file_type in file_name]

def get_date_from_str(str):
    return str[:4] + "/" + str[4:6] + "/" + str[6:]

def get_from_file(article, date_str, isFrance=False):
    with open(article, 'r') as f:
        text = l.json.load(f)["contentAsText"] if isFrance else "\n".join(f.readlines())  
        date = get_date_from_str(date_str)
        return [date, text]

def get_ocr(article, date_str, withHeight=False):
    date = get_date_from_str(date_str)
    root = l.ET.parse(article).getroot()
    txt = ""
    height = []
    for textblock in root.findall(".//{http://www.loc.gov/standards/alto/ns-v2#}TextBlock"):
        for s in textblock.findall('.//{http://www.loc.gov/standards/alto/ns-v2#}String'):
            txt += s.attrib["CONTENT"] + " "
            if withHeight:
                to_append = int(s.attrib["HEIGHT"]) if "HEIGHT" in s.attrib else -1
                height.append(to_append)
    row = [date, txt, height] if withHeight else [date, txt]
    return row

def extract_files(file, country_id, extract):
    country = country_id[0]
    id = country_id[1]
    rows = []

    print("(" + country + ") On file " + "/".join(file.split("/")[2:]))
    if extract:
        start_extraction = l.time.time()
        if country == "USA":        
            folder = l.tarfile.open(file)
            folder.extractall('./Institution (2)/United States/' + str(id))
        else :
            with l.zipfile.ZipFile(file) as zip_ref:
                zip_ref.extractall('./Institution (2)/' + country + "/" + str(id))

        print("Extraction done, took " + U.get_time(start_extraction))

    pattern = './Institution (2)/United States/' + str(id) + U.DATA_PATTERNS[country] if country == "USA" else './Institution (2)/' + country + "/" + str(id) + U.DATA_PATTERNS[country] 
    every_article = l.glob.iglob(pattern, recursive = True)

    start_process = l.time.time()
    for i, article in enumerate(every_article):

        if i % 1000 == 0:
            print("(" + country + ") At article " + str(i))
        if country == "USA":
            date_str = "".join(article.split("/")[5:8])
            rows.append(get_ocr(article, date_str, withHeight=True))
        elif country == "France":
            rows.append(get_from_file(article, article.split("/")[5], isFrance=True))
        elif country == "Hamburg":
            date_str = article.split("/")[-2][-8:]
            rows.append(get_ocr(article, date_str))
        elif country == "Spain":
            date_str = article.split("/")[4][:8]
            rows.append(get_from_file(article, date_str))

    print("Processing done, took " + U.get_time(start_process))
    columns = ["date", "text"] if country == "France" or country == "Spain" else ["date", "text", "height"]
    df = l.pd.DataFrame(rows, columns=columns)
    df_name = "processed/" + C.DF_NAMES[C.COUNTRY_TO_LANG[country]]+ "_" + str(id) + ".zip" if country == "USA" or country == "Hamburg" else U.DF_NAMES[U.COUNTRY_TO_LANG[country]] + ".zip"
    U.compress(df, df_name)

def merge_dfs(country, num_of_splits):
    frames = []
    for i in range(0,num_of_splits):
        df_name = "processed/" + C.DF_NAMES[C.COUNTRY_TO_LANG[country]]+ "_" + str(i) + ".zip"
        df_i = l.pd.read_csv(df_name)
        frames.append(df_i)
    return l.pd.concat(frames)


n = len(l.sys.argv)
possible_countries = ["USA", "France", "Hamburg", "Spain"]
extract = False
delete = False

if l.sys.argv[-1] == "-extr":
    extract = True
    n -= 1

if(n > 2) and l.sys.argv[-2] == "-del":
    delete = True
    n -= 1

all_files = l.tarfile.open(C.RAW_DATA_PATH)

for i in range(1, n):
    country = l.sys.argv[i]
    if country not in possible_countries:
        print("This country doesn't exist : " + country)
        continue
    print("Starting work on " + country + " !")
    file_type = "tar" if country == "USA" else "zip"
    files = get_country_files(country, file_type, all_files)
    if extract:
        for file in files:
            l.subprocess.run(["tar", "-xvf", U.RAW_DATA_PATH, file], stdout=l.subprocess.PIPE, text=True) 

    rows = []
    if country == "France" or country == "Spain":
       extract_files(files[0], (country, 0))
    else:
        l.subprocess.run(["mkdir", "processed"])
        num_processes = 16 if country == "USA" else 3
        args = zip(files*num_processes, zip([country]*num_processes,list(range(0, num_processes))))
        with l.Pool(num_processes) as p:
            p.starmap(extract_files, args)
        full_df = merge_dfs(country, num_processes)
        if delete:
            l.subprocess.run(["rm", "-rf", "processed"])
        U.compress(full_df, C.DF_NAMES[C.COUNTRY_TO_LANG[country]] + ".zip")
        
    if delete :
        l.subprocess.run(["rm", "-rf", "Institution (2)"])
            
