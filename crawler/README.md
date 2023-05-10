# DailyHunt Crawler

This crawler scrapes news articles from DailyHunt.

To recreate the dataset:
- Download files from the [bucket](https://console.cloud.google.com/storage/browser/varta-eu/data-release)
- Set the `INFILE` in `settings.py` to the path of the file you want to recreate; change `OUTFILE` to the path where you want the data to be saved
- The output file will contain the following keys:
    - `id`: unique identifier of the article in the format of "nxxxxxxxxx"
    - `headline`: headline of the article
    - `text`: the main content of the article
    - `url`: the DailyHunt url of the article
    - `source_media`: name of the publisher that DailyHunt aggregates this article from
    - `source_url`: the url of this article from the original publisher
    - `publication_date`: timestamp 
    - `tags`: a list of categories that the article belongs to
    - `reactions`: a dictionary of the reactions from the readers
    - `word_count`: word count based on white space delimiter
    - `langCode`: language code based on two-digit ISO 639-1 convention

## Running the code
The code is based on Scrapy and BeautifulSoup. Install them both
```
conda activate <env_name>
conda install -c conda-forge scrapy
conda install -c anaconda beautifulsoup4
```
and run
```
scrapy crawl dh
```
