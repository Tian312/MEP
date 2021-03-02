
# coding: utf-8

# In[ ]:


# coding=UTF-8
# http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&WebEnv=NCID_1_196289963_130.14.18.34_9001_1524513200_1515294449_0MetA0_S_MegaStore&query_key=1&retmode=json&rettype=abst    ract


# In[ ]:

import os,sys
import csv
import re
import urllib.request as ur
import os,codecs
from time import sleep

from xml.dom import minidom 
import xml.etree.ElementTree as xml_parser

import urllib.parse
import nltk


# In[ ]:


def clean_title(title):
    
    title_new = re.sub("\s+$","",title)
    title_new = re.sub("^\s+","",title_new)
    title_new = re.sub("[^\w]\s?$","",title)
    title_new = re.sub("\s+","%20",title_new)
    return title_new


# In[ ]:


def identify_ascii(string):
    1


# In[ ]:


# terms = ["Gender%20differences%20in%20obtaining%20and%20maintaining%20patent%20rights[Title]"]
# terms =["cancer[mesh]+AND+2017:2018[pdat]+english[Language]"]


def get_abstract(title):
    
    term = clean_title(title)
    #print (term)
    pmid_text = ""
    title_text = ""
    abstract_text = ""
    
    baseURL = 'http://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
    eutil = 'esearch.fcgi?'
    dbParam = 'db=pubmed'
    usehistoryParam = '&usehistory=y'
    rettype = '&rettype=json'
    eutilfetch = 'efetch.fcgi?'
    fieldParam = '&field=title'
    termParam = '&term='+ str(term)
    
    retmax = 500
    retstart = 0
    #print ("...url", baseURL+eutil+dbParam+termParam+usehistoryParam)
    fullURL = baseURL+eutil+dbParam+fieldParam+termParam+usehistoryParam+rettype
 
    # =====urllib.error.HTTPError: HTTP Error 400: Bad Request======== dealing with non-ascii url =====
    try:

        f = ur.urlopen(fullURL)

    except:
    #try:

        fullURL = urllib.parse.quote(fullURL,safe=':/') # <- here
        f = ur.urlopen(fullURL)

    #f = ur.urlopen(fullURL)
    data = f.read().decode('utf-8')
    #print (data)
    try:
        webenv = "&WebEnv=" + re.findall ("<WebEnv>(\S+)<\/WebEnv>", data)[0]
    except:
        return pmid_text, title_text, abstract_text
    count = int(re.findall("<Count>(\d+?)</Count>",data)[0])
    if count == 0:
        print ("Warning: no file found for '", title,"'" )
        return pmid_text, title_text, abstract_text

        

    querykey = "&query_key=" + re.findall("<QueryKey>(\d+?)</QueryKey>",data)[0]
#    print ("\n====",querykey,"===\n",webenv,"\n")
    rettype = "&rettype=abstract"
    str_retmax = "&retmax=" + str(retmax)
    #retmode = "&retmode=text"
    retmode ="&retmode=xml"
    
    str_retstart = "&retstart=" + str(retstart)
    fetch_url = baseURL+eutilfetch+dbParam+querykey+webenv+str_retstart+str_retmax+retmode+rettype

    
    #print ("---fetch url", fetch_url)

    sleep(5)
    
    
    article_info = {}
    # [pmid] = journal_title|article_title|abstract
    
    
    try:
        #Open the webpage with the fetch utility.

        fetch = ur.urlopen(fetch_url)
        datam = fetch.read().decode('utf-8') # xml of one article
        #print (datam)
        
        datam = re.sub("<i>","", datam)  # remove <i> in xml to avoid failure in parsing .text
        datam = re.sub("</i>","", datam)
        datam = re.sub("<sup>", "", datam)  # remove <i> in xml to avoid failure in parsing .text
        datam = re.sub("</sup>", "", datam)
        datam = re.sub("<sub>", "", datam)  # remove <i> in xml to avoid failure in parsing .text
        datam = re.sub("</sub>", "", datam)
        datam = re.sub("<em>", "", datam)  # remove <i> in xml to avoid failure in parsing .text
        datam = re.sub("</em>", "", datam)
        datam = re.sub("<strong>", "", datam)  # remove <i> in xml to avoid failure in parsing .text
        datam = re.sub("</strong>", "", datam)
        datam = re.sub("<b>", "", datam)  # remove <b> in xml to avoid failure in parsing .text
        datam = re.sub("</b>", "", datam)

        xmldoc = xml_parser.fromstring(datam)
        #print (xmldoc)
        articles = xmldoc.findall("PubmedArticle")

        if articles is not None:
            for article in articles:
                data = article.find("MedlineCitation")
                should_not_stop = re.match('^$',pmid_text)
                if should_not_stop:
                    1
                else:
                    break
                if data is not None:
                    Article = data.find("Article")
                    if Article is not None:
                        title = Article.find("ArticleTitle")
                        if title is not None:
                            title_text = title.text
                            #print (title_text)
                            # filter non-matched titles --------
                            title_words = re.split("\s+",title_text)

                            title_words[-2] = re.sub('^\W','',title_words[-2])
                            title_words[-2] = re.sub('\W$', '', title_words[-2])
                            title_words[-3] = re.sub('^\W', '', title_words[-3])
                            title_words[-3] = re.sub('\W$', '', title_words[-3])
                            title_words[1] = re.sub('^\W', '', title_words[1])
                            title_words[1] = re.sub('\W$', '', title_words[1])
                            title_words[2] = re.sub('^\W', '', title_words[2])
                            title_words[2] = re.sub('\W$', '', title_words[2])
                            title_words[3] = re.sub('^\W', '', title_words[3])
                            title_words[3] = re.sub('\W$', '', title_words[3])

                            match_backward_1 = re.search(title_words[-2], term, re.I)
                            match_backward_2 = re.search(title_words[-3], term, re.I)
                            match_forward_1 = re.search(title_words[1], term, re.I)
                            match_forward_2 = re.search(title_words[2], term, re.I)
                            match_forward_3 = re.search(title_words[3], term, re.I)

                            if (match_backward_1 is None) or  (match_backward_2 is None) or (match_forward_1 is None) or (match_forward_2 is None) or (match_forward_3 is None):
                                #print (title_words[-2],title_words[-3],title_words[1],title_words[2],title_words[3])
                                #print (term)
                                pmid_text = ""
                                title_text = ""
                                abstract_text = ""

                                continue
                            
                            #====================================
                            
                        abstract = Article.find("Abstract")
                        if abstract is not None:
                            for seg in abstract:
                                try:
                                    abstract_text = abstract_text + "\n"+seg.attrib['NlmCategory']+":"+" "+seg.text
                                except:
                                    abstract_text = abstract_text+"\n"+seg.text
                    
                    pmid = data.find("PMID")
                    if pmid is not None:
                        pmid_text = pmid.text
                    
        abtract_text = re.sub("^\s+","",abstract_text)
        ab_groups = re.split("\n",abstract_text)
        abstract_text_new =""
        for a in ab_groups:
            if a == "":
                continue
            c = re.search("©|(copyright)", a)
            if c:
                continue
            abstract_text_new=abstract_text_new+" "+a

        return pmid_text, title_text, abstract_text_new
        
        
        
    except:
        print ("!!!!!!!!!!! failure   !!!!!!",title )
        return pmid_text, title_text, abstract_text
        
    
                
def get_abstract_bypmid(pmid):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id="+str(pmid)+"&retmode=XML&rettype=abstract"

    try:
        fetch = ur.urlopen(url)
        datam = fetch.read().decode('utf-8')  # xml of one article
    except:
        return None

    datam = re.sub("<i>", "", datam)  # remove <i> in xml to avoid failure in parsing .text
    datam = re.sub("</i>", "", datam)
    datam = re.sub("<sup>", "", datam)  # remove <i> in xml to avoid failure in parsing .text
    datam = re.sub("</sup>", "", datam)
    datam = re.sub("<sub>", "", datam)  # remove <i> in xml to avoid failure in parsing .text
    datam = re.sub("</sub>", "", datam)
    datam = re.sub("<em>", "", datam)  # remove <i> in xml to avoid failure in parsing .text
    datam = re.sub("</em>", "", datam)
    datam = re.sub("<strong>", "", datam)  # remove <i> in xml to avoid failure in parsing .text
    datam = re.sub("</strong>", "", datam)
    datam = re.sub("<b>", "", datam)  # remove <b> in xml to avoid failure in parsing .text
    datam = re.sub("</b>", "", datam)

    xmldoc = xml_parser.fromstring(datam)
    articles = xmldoc.findall("PubmedArticle")
    abstract_text = ""
    title_text=""
    if articles is not None:
        for article in articles:
            data = article.find("MedlineCitation")
            if data is not None:
                article = data.find("Article")
                if article is not None:
                    title = article.find("ArticleTitle")
                    
                    #print("Title:",title.text)
                    title_text = title.text
                    abstract = article.find("Abstract")
                    if abstract is not None:
                        for seg in abstract:
                            try:
                                abstract_text = abstract_text + "\n" + seg.attrib['NlmCategory'] + ":" + " " + seg.text
                                
                            except:
                                abstract_text = abstract_text + "\n" + seg.text
                                #print(seg.text)

    abtract_text = re.sub("^\s+", "", abstract_text)

    ab_groups = re.split("\n", abstract_text)
    abstract_text_new = ""
    for a in ab_groups:
        c = re.search("©|(copyright)", a)
        if a == "":
            continue
        elif c:
            continue
        else:
            a = re.sub("^\s+","",a)
            a = re.sub("\s+^","",a)
            if abstract_text_new != "":
                abstract_text_new = abstract_text_new + " " + a
            else:
                abstract_text_new = a
           
    return title_text,abstract_text_new

import json
def get_abstract_bypmcid(pmcid):
    #if not re.search("PMC",pmcid):
    #    pmcid= "PMC"+str(pmcid)
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pmc&id="+str(pmcid)+"&retmode=JSON"
    # https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=3159986&retmode=XML&rettype=abstract
    # https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pmc&id=3159986&retmode=JSON&rettype=abstract
    #try:
    fetch = ur.urlopen(url)

    datam = json.loads(fetch.read().decode('utf-8'))  # xml of one article
    #except:
    #    return None
    ids = datam["result"][pmcid]["articleids"]
    pmid = ""
    for id in ids:
        if "pmid" in id.values():
            pmid= id["value"]
       
    return get_abstract_bypmid(pmid)

def tokenize(rawtext):
    sents = nltk.sent_tokenize(rawtext)
    parsed_sents=[]
    for s in sents:
        parsed_s =" ".join(nltk.word_tokenize(s))
        parsed_sents.append(parsed_s)
    parsed_text = " ".join(parsed_sents)
    return parsed_text 



index = 0
exceptionlist=[]

pmidlist= codecs.open(sys.argv[1]) #pmid-2015-2019
#collection = codecs.open(sys.argv[2],'w')
ex_path = os.path.join(sys.argv[2],"exceptionlist")
print(ex_path)
if not os.path.exists(sys.argv[2]):
    os.mkdir(sys.argv[2])
exceptionlistfile=codecs.open(ex_path,'w')

try:
    if sys.argv[3] and sys.argv[3] not in ["PMCID","PMID","pmcid","pmid","pmc","PMC"] :
        sys.exit("Invalid id type. Please try to select one of the following: PMCID, PMID (default PMID). ")
    elif sys.argv[3] and re.search("pmc",sys.argv[3],re.I):
        id_type = "PMCID"
except:
    id_type="PMID"

for id in pmidlist:
    id = id.rstrip()
    filename=id+".txt"
    
    if re.search("^#",id):
        continue 
    if re.search("^pmid", id):
        continue
    if re.search("^\s*$",id):
        continue
    id = id.rstrip().split("\t")[0]
    print (id)
    index +=1
    if index **500 == 0:
        print(index,"finished...")
        sleep(60)
    elif index ** 100 == 0 :
        sleep(30)
        print (index, "finished...")
    elif index ** 10 == 0:
        sleep(5)
    else:
        sleep(2)
    try:
        if id_type == "PMID":
            title, ab = get_abstract_bypmid(id)
        if id_type == "PMCID":
            title, ab = get_abstract_bypmcid(id)
        if ab == "":
            print ("no abstract")
            continue

        outfile = codecs.open(os.path.join(sys.argv[2],filename),'w')
        outfile.write(str(title)+ " "+ab+"\n")
        #collection.write(str(id)+" | "+str(title)+" | "+ab+"\n")
    
    except:
        print("Expection at ", id,"!!!")
        exceptionlistfile.write(str(id)+"\n")
        continue


