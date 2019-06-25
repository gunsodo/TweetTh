# scrape Tweets
scrapy crawl TweetScraper -a query="#RIPThailand" -a lang="th"

# extract text and save to file
python utils.py --file_dir /data/tweet --path extract.txt