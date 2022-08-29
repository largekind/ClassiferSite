#!/usr/bin/env python
# -*- coding: utf-8 -*-
import secrets
from tkinter import W
from flickrapi import FlickrAPI
from urllib.request import urlretrieve
from dotenv import load_dotenv
import os , sys , time

# 環境変数ファイル読み込み
dotenv_path = './.env'
load_dotenv(dotenv_path)
# 環境変数設定
key = os.environ.get("FLICKR_KEY").encode('UTF-8')
secret = os.environ.get("FLICKR_SECRET").encode('UTF-8')

wait_time = 1 #待機時間
# 検索ワード取得、保存用ディレクトリ作成
keyword = sys.argv[1]
savedir = "./data/" + keyword + '/'
os.makedirs(savedir,exist_ok=True)

# flickerインスタンス作成、画像検索
flicker = FlickrAPI(key, secret, format='parsed-json')

result = flicker.photos.search(
  text = keyword,
  per_page = 400,
  media = 'photos',
  sort = 'relevance',
  safe_search = 1,
  extras = 'url_q, license',
)

# 画像データの抽出
photos = result['photos']

for i , photo in enumerate(photos['photo']):
  url_q = photo['url_q']
  filepath = savedir +  photo['id'] + '.jpg'
  if os.path.exists(filepath): continue #既にある場合はスキップ

  urlretrieve(url_q, filepath) #url取得してfilepathに保存
  time.sleep(wait_time)

