#Below code is used to scrap the players characteristics (height, birthdate, hand, backhand...) that are present in the ATP website player's dedicated profile page.

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
from webdriver_manager.chrome import ChromeDriverManager
import csv

fileoutput  = r'.\playersc.csv'
url = 'https://www.atptour.com/en/players/aplayer/XXXX/overview'

# Inserting the scope
code_list = ['mn13','bk92','sf89','bm95','a853','e698','ch05','sn77','g806','mg54','y218']

code_list = list(set(code_list))

players_list = []
for count, item  in  enumerate(code_list):
    fname,lname,rank,height,bday,hand, back_hand,country,weight = 'NULL','NULL','NULL','NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL'
    url_var = url.replace('XXXX',item)    
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.get(url_var)
    time.sleep(10)
    lname_w = driver.find_element_by_xpath('//*[contains(concat( " ", @class, " " ), concat( " ", "last-name", " " ))]')
    lname = lname_w.text
    fname_w = driver.find_element_by_xpath('//*[contains(concat( " ", @class, " " ), concat( " ", "first-name", " " ))]')
    fname = fname_w.text
    try:
        bday_w = driver.find_element_by_xpath('//*[contains(concat( " ", @class, " " ), concat( " ", "table-birthday", " " ))]')
        bday = bday_w.text
    except:
        bday = 'NULL'
    rank_w = driver.find_element_by_xpath('//*[contains(concat( " ", @class, " " ), concat( " ", "data-number", " " ))]')
    rank = rank_w.text
    try:
        height_w = driver.find_element_by_xpath('//*[contains(concat( " ", @class, " " ), concat( " ", "table-height-cm-wrapper", " " ))]')
        height = height_w.text
    except : height = 'NULL'
    try:
        search_hands = WebDriverWait(driver,100).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".table-value")))
        hands = search_hands[1].text 
        hand, back_hand = hands.split(',')[0] , hands.split(', ')[1]
    except: hand, back_hand = 'NULL' , 'NULL'
    try:
        cou_w = driver.find_element_by_xpath('//*[contains(concat( " ", @class, " " ), concat( " ", "player-flag-code", " " ))]')
        cou = cou_w.text
        cou = cou.replace(" ", "")
        country = cou
        # country = cou[cou.rfind('\n')+1:]
    except: country = 'NULL'
    try:
        weight_w = driver.find_element_by_xpath('//*[contains(concat( " ", @class, " " ), concat( " ", "table-weight-lbs", " " ))]')
        weight = weight_w.text
    except: weight = 'NULL'
    
    driver.close()
    player_list = [item,fname,lname,rank,height,bday,hand, back_hand,country,weight]
    players_list.append(player_list)
    print(count)
cou[cou.rfind('\n')+1:]
df = pd.DataFrame(players_list)
df.columns =  ['ATP_code','fname','lname','rank','height','bday','hand','back_hand', 'country','weight']
df['height'] = df['height'].str.replace('''cm\)''','',regex = True)
df['height'] = df['height'].str.replace('\(','',regex = True)
df['bday'] = df['bday'].str.replace('\)','',regex = True)
df['bday'] = df['bday'].str.replace('\(','',regex = True)
df['bday'] = df['bday'].str.replace('.','',regex = True)

df.to_csv(fileoutput, index = False)
