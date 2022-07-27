# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 13:28:08 2021

@author: hgobb
"""

from selenium import webdriver
import time
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import csv
import importlib.util
import sys
from datetime import date, timedelta

### setup
# input
url = 'https://www.atptour.com/en/rankings/singles'
url1_prefix = 'https://www.atptour.com/en/rankings/singles?rankDate='
# output
fileoutput  = r'C:\Users\hgobb\Documents\Aleph\tennis_data\ranking.csv'
### params
rank_from = 1
rank_to = 1000
#upload to aleph yes 1 / no 0
upload_to_aleph = 0

###start
#get ranking date
now = date.today()
a_date = str(now - timedelta(days = now.weekday()))


# driver = webdriver.Chrome(ChromeDriverManager().install())
# driver.get(url)
# time.sleep(10)
# x_date = driver.find_element_by_xpath('//*[contains(concat( " ", @class, " " ), concat( " ", "dropdown-label", " " ))]')
# a_date = x_date.text
# x_date1 = driver.find_element_by_xpath('//*[contains(concat( " ", @class, " " ), concat( " ", "current", " " ))]')
# a_date1 = x_date1.text
# aaa = BeautifulSoup(x_date1.text, 'html.parser').a.attrs
# soup = BeautifulSoup(x_date1.text, 'html.parser')
# print(soup.select_one('._3e12V').text)
# print(a_date1)

# now = date.today()



# driver.close()

#get rankings
url1 = url1_prefix +a_date +'&rankRange=' + str(rank_from) + '-' + str(rank_to)
driver = webdriver.Chrome(ChromeDriverManager().install())
driver.get(url1)
time.sleep(10)

ranklist = [['a_date','rank','points' , 'name', 'web_page']]
for k in range(1,rank_to+1):
    playdet = []
    
    rnk = driver.find_element_by_xpath("//table/tbody/tr["+str(k)+"]/td["+str(1)+"]").text
    pnt = driver.find_element_by_xpath("//table/tbody/tr["+str(k)+"]/td["+str(6)+"]").text
    aname = driver.find_element_by_xpath("//table/tbody/tr["+str(k)+"]/td["+str(4)+"]").text
    html = driver.find_element_by_xpath("//table/tbody/tr["+str(k)+"]/td["+str(4)+"]").get_attribute('outerHTML') 
    attrs = BeautifulSoup(html, 'html.parser').a.attrs
    lnk = attrs['href']
    
    playdet = [a_date,rnk,pnt,aname,lnk]
    
    ranklist.append(playdet)
    
driver.close()    

with open(fileoutput, 'w', newline='') as myfile:
    wr = csv.writer(myfile)
    for sublist in ranklist:
        wr.writerow(sublist)
if upload_to_aleph == 1:
    sys.path.append(r'C:\Users\hgobb\Documents\Aleph\Scripts')
    import atp_rank_scrap_loader
    atp_rank_scrap_loader.rank_loader()


