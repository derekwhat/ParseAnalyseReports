#!/usr/bin/env python
# coding: utf-8

# In[13]:


from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options
from selenium.common import exceptions
from requests import get
from requests.exceptions import RequestException
from contextlib import closing

from bs4 import BeautifulSoup
import pandas as pd
import re
import os
from datetime import datetime as dt, date
from itertools import chain
import time


# In[4]:


#source: https://realpython.com/python-web-scraping-practical-introduction/
def simple_get(url):
    """
    Attempts to get the content at `url` by making an HTTP GET request.
    If the content-type of response is some kind of HTML/XML, return the
    text content, otherwise return None.
    """
    try:
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None

    except RequestException as e:
        log_error('Error during requests to {0} : {1}'.format(url, str(e)))
        return None

def is_good_response(resp):
    """
    Returns True if the response seems to be HTML, False otherwise.
    """
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200
            and content_type is not None
            and content_type.find('html') > -1)

def log_error(e):
    """
    It is always a good idea to log errors.
    This function just prints them, but you can
    make it do anything.
    """
    print(e)


#https://stackoverflow.com/questions/26494211/extracting-text-from-a-pdf-file-using-pdfminer-in-python
from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

def convert_pdf_to_txt(path, maxpages=0):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    caching = True
    pagenos=set()

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=False):
        interpreter.process_page(page)

    text = retstr.getvalue()

    fp.close()
    device.close()
    retstr.close()
    return text


# In[5]:


def cmap(f, x):
    return chain.from_iterable(map(f, x))

def ref_date_jp(rpt_html_url):
    a = BeautifulSoup(simple_get(rpt_html_url), 'html.parser').find_all("td")[3].text[6:]
    a = cmap(lambda x: x.split('年'), [a])
    a = cmap(lambda x: x.split('月'), a)
    a = cmap(lambda x: x.split('日'), a)
    a = list(a)
    return dt(int(a[0]),int(a[1]),int(a[2]))


# In[7]:


def LD(s, t):
    if s == "":
        return len(t)
    if t == "":
        return len(s)
    if s[-1] == t[-1]:
        cost = 0
    else:
        cost = 1

    res = min([LD(s[:-1], t)+1,
               LD(s, t[:-1])+1,
               LD(s[:-1], t[:-1]) + cost])
    return res

months = "jan,feb,mar,apr,may,jun,jul,aug,sep,oct,nov,dec".split(',')

def match(jurai):
    y = map(lambda x: LD(x, jurai), months)
    y = map(lambda x: (x[1],x[0]), enumerate(y))
    y = sorted(y)
    return months[next(iter(y))[1]]


# In[8]:


def save_rpt_eng(url,tgt):
    file_url = get(url)
    with open(tgt, 'wb')as f:
        f.write(file_url.content)
        f.close()

def jury(date):
    if not date:
        return date
    date = date.replace(r'\xa0', r' ')
    date = ' '.join(date.split(','))

    p = re.compile(r'[\w]+', re.IGNORECASE)
    date = p.findall(date)
    date = date[-3:]

    if re.match(r'\d+', date[0]) is not None:
        date = [date[1],date[0],date[2]]
#         print(date)
    date[0] = date[0][:3].lower()
    if date[0] == "jur":
        date[0] = "jul"
    date[0] = match(date[0])
    date = " ".join(date)
    return date

def get_ref_date_eng(rpt_path):
    result = convert_pdf_to_txt(rpt_path,maxpages=1).strip()[:800]
    regex = r'revi.{0,50}[\d]{4}\W{0,50}\s|update.{0,50}[\d]{4}\W{0,50}\s'
    p = re.compile(regex, re.IGNORECASE)
    ref_date_eng = p.findall(result)

    #try again
    if ref_date_eng == []:
        p = re.compile(r':*.{0,50}[\d]{2,4}\W{0,50}\s', re.IGNORECASE)
        ref_date_eng = p.findall(result)[0]
    else:
        ref_date_eng = p.findall(result)[0]

    #parse out date
    i = ref_date_eng.find(':')
    if i == -1:
        i = ref_date_eng.find('：')
    ref_date_eng = ref_date_eng[i+1:].strip()
    ref_date_eng = jury(ref_date_eng)

    return ref_date_eng

def ref_date_eng(src,rpt_path, keep_saved=False):
    save_rpt_eng(src,rpt_path)
    dt = get_ref_date_eng(rpt_path)
#     if keep_saved is False:
#         os.remove(rpt_path)
    return dt


# In[9]:


def parse_jp_data(tr_list, root, col_names):
    jp = [[ref_date_jp(root+row.find_all("td")[7].find_all("a")[1]['href'][5:]),
           row.find_all("td")[6].text.strip(),
           '{}{}'.format(root,row.find_all("td")[8].find_all("input")[0]['onclick'].split()[-1][1:-3])]
          for row in tr_list]
    jp_data = pd.DataFrame(data=jp, columns=col_names)
    return jp_data

def parse_eng_data(tr_list, root, path, col_names):
    eng = [[ref_date_eng('{}{}'.format(root,row.find_all("a")[0]['href'][5:]),
                         '{}\{}'.format(path,row.find_all("a")[0]['href'][12:])),
            row.find_all("td")[0].text.strip(),
            '{}{}'.format(root,row.find_all("a")[0]['href'][5:])]
           #skip the reports which say 'correction', not relevant to getting english rpt upload dates
           if row.find_all("a")[0].text.strip().lower().find('correction') is -1 else []
           for row in tr_list]
    eng_data = pd.DataFrame(data=eng, columns=col_names)
    return eng_data


# In[11]:


opts = Options()
# opts.headless = True # Operating in headless mode
browser = Firefox(options=opts)

company_file = r'Companies who submitted English CG Reports - 20191230.xlsx'
path = r'C:\Users\Derek Ho\Desktop\Uni\UW 4th year W18, W19 + TiTech\2019 Exchange TiTech\Q3Q4 Research seminar\Research Project'
companies = pd.read_excel('{}\{}'.format(path,company_file)).dropna()
site = r'https://www2.tse.or.jp/tseHpFront/CGK020010Action.do'
root = r"https://www2.tse.or.jp/disc"
dump_path = '{}{}'.format(os.getcwd(),'\CG_reports_eng')
cols_jp = ['ref_date','upload_date_jp','xbrl_src']
cols_eng = ['ref_date', 'upload_date_eng','pdf_src']
cols = ['ref_date', 'upload_date_jp', 'xbrl_src', 'upload_date_eng', 'pdf_src', 'company_code']
companies_data = pd.DataFrame(columns=cols, dtype='object')
company_codes = companies['Code']


# In[12]:


start = dt.now()

blacklist = [9001.0, 9422.0, 4298.0]
#tobu railway*, conexio corporation*,Proto Corporation*
#*: deemed too hard to ger programmatically for reasons
#   - date does not fit all on one line (human error on the company side)
#   - file is an image, no text
#   - file is converted to non-text
company_codes = list(filter((lambda x: x not in blacklist), company_codes))
n = len(company_codes)
i = 0
for code in company_codes:
    browser.get(site)

    #search company
    search_form = browser.find_element_by_name('eqMgrCd')
    company_code = str(int(code))+"0"
    search_form.send_keys(company_code)
    browser.execute_script("submitCG()")

    #get company profile
    time.sleep(2)
    # lazy way to wait for page to load.. should wait for event instead
    try:
        browser.find_element_by_link_text(company_code).click()
    except exceptions.NoSuchElementException:
        browser.refresh()
        time.sleep(2)
        browser.find_element_by_link_text(company_code).click()

    #get tables of JP and ENG data
    time.sleep(2)
    try:
        soup = BeautifulSoup(browser.page_source,'html.parser')
        table_eng = soup.find_all("h4")[4].fetchNextSiblings()[0]
        table_jp = soup.find_all("h4")[5].fetchNextSiblings()[0]
    except exceptions.NoSuchElementException:
        browser.refresh()
        time.sleep(2)
        soup = BeautifulSoup(browser.page_source,'html.parser')
        table_eng = soup.find_all("h4")[4].fetchNextSiblings()[0]
        table_jp = soup.find_all("h4")[5].fetchNextSiblings()[0]

    #parse JP and ENG tables
    jp_data = parse_jp_data(table_jp.find_all("tr")[3:], root, cols_jp)
    eng_data = parse_eng_data(table_eng.find_all("tr")[2:], root, dump_path, cols_eng)

    #format column for joining
    time.sleep(1)#<-- can probably remove this
    eng_data['ref_date'] = pd.to_datetime(eng_data.ref_date, errors='ignore')
#     eng_data['ref_date'] = eng_data['ref_date'].dt.strftime('%Y/%m/%d')
    jp_data['ref_date'] = pd.to_datetime(jp_data.ref_date)
#     jp_data['ref_date'] = jp_data['ref_date'].dt.strftime('%Y/%m/%d')

    #create table for company
    comp_data = jp_data.merge(eng_data,how='left', left_on='ref_date', right_on='ref_date')
    comp_data['company_code'] = company_code

    companies_data = companies_data.append(comp_data)
    i += 1
    print('{}/{} parsed!'.format(i,n))
    companies_data.to_csv('{}\{}{}.csv'.format(dump_path,'disclosure_times',date.today()))

end = dt.now()
deltat = end - start
#for good measure
companies_data.to_csv('{}\{}{}.csv'.format(dump_path,'disclosure_times',date.today()))
print('{} or {} to finish parsing through companies.'.format(deltat.seconds, deltat.seconds/60))
print('{} or {} on average for each company.'.format(deltat.seconds/n, deltat.seconds/60/n))
browser.close()




# In[152]:


f = '140120161012411241.pdf'
p = '{}\{}'.format(r'C:\Users\Derek Ho\Desktop\Uni\UW 4th year W18, W19 + TiTech\2019 Exchange TiTech\Q3Q4 Research seminar\Research Project\CG_Reports_eng',
                   f)
# get_ref_date_eng(r'C:\Users\Derek Ho\Desktop\Uni\UW 4th year W18, W19 + TiTech\2019 Exchange TiTech\Q3Q4 Research seminar\Research Project\CG_Reports_eng\140120191220439175.pdf')
result = convert_pdf_to_txt(p, maxpages=1).strip()[:800]

# p = re.compile(r':*.{0,50}[\d]{2,4}\W{0,50}\s', re.IGNORECASE)
p = re.compile(r'revi.{0,50}[\d]{4}\W{0,50}\s|update.{0,50}[\d]{4}\W{0,50}\s', re.IGNORECASE)
r = p.findall(result)
r[0]

jury(r[0])
# t = pd.DataFrame([jury(r[0])],columns=['test'])
# t['test'] = pd.to_datetime(t.test, errors='ignore')
# t


# In[116]:


companies_data.tail(50)
