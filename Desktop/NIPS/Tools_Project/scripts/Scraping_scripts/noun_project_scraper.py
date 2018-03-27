import urllib
from bs4 import BeautifulSoup
import pandas as pd
import requests
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import numpy as np
import sys
import io
import os
import urllib.request
sys.path.append('/data/graphics/graphsProject/graphScraping/geckodriver/')

SCRAPE_TARGET_FILE = sys.argv[1]

icon_url_root = "https://thenounproject.com/icon/"
search_url_root = "https://thenounproject.com/search/?q="

def parse_query(tag,save_dir):
    search_url = search_url_root+tag
    driver = webdriver.Chrome()
    driver.get(search_url)
    source = driver.page_source
    pages_viewed = 0
    time.sleep(3)
    # Scroll down 10 pages
    for i in range(500):
        driver.execute_script('window.scrollTo(0,document.body.scrollHeight);')
    source = driver.page_source
    driver.quit()
    soup = BeautifulSoup(source, 'html.parser')
    all_icons = soup.find_all('div', class_='Grid-cell loaded')
    print('Downloading %s icons'%len(all_icons))
    for icon in all_icons:
        icon_ID = icon.a.get('href').split('/')[-1]
        icon_url = icon.img.get('src')
        fname = "%s/%s_%s.jpg"%(save_dir,tag,icon_ID)
        print(fname)
        urllib.request.urlretrieve(icon_url,fname)

f = open(os.getcwd() + '/' + SCRAPE_TARGET_FILE,'r')
tag_list = [c.rstrip() for c in f.readlines()]
f.close()

for tag in tag_list:
    if len(tag.split(' ')) > 1:
        search_tag = '%20'.join(tag.split())
        save_tag = '_'.join(tag.split())
    else:
        search_tag = tag
        save_tag = tag
    #create if directory doesn't exist
    if os.path.isdir(os.getcwd() + '/Icons/' + save_tag):
        save_dir = os.getcwd() + '/Icons/' + save_tag
        pass
    else:
        save_dir = os.getcwd() + '/Icons/' + save_tag
        os.mkdir(save_dir)
        print('Created directory for %s'%tag)
    parse_query(search_tag,save_dir)
