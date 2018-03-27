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
sys.path.append('/data/graphics/graphsProject/graphScraping/geckodriver/')
TOTAL_IDS = 1304232 #MAX NUMBER OF ICONS IN THE NOUN PROJECT
icon_url_root = "https://thenounproject.com/icon/"
search_url_root = "https://thenounproject.com/search/?q="
COLS = ['ID', 'NAME', 'TAGS', 'IMG_URL']


tag_list = ['tool']



def parse_query(tag):
	earch_url = search_url_root+tag
	driver = webdriver.Chrome()
	driver.get(search_url)
	source = driver.page_source
	lastHeight = driver.execute_script("return document.body.scrollHeight")
	pages_viewed = 0

parse_query('tool')

