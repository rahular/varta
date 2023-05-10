import json
import scrapy
from pathlib import Path
from bs4 import BeautifulSoup as bs
from scrapy.utils.project import get_project_settings

class DHSpider(scrapy.Spider):
    name = "dh"

    def __init__(self):
        self.save_dir = Path()
        self.ids = self.get_id_list()
        self.base_url = "https://pwa-api.dailyhunt.in/api/v100/moredetails?morecontenturl=http://api-news.dailyhunt.in/api/v2/posts/article/content/{}?useWidgetPosition=false&feedConfigKey=HASHTAG_CHRONO&ignoreTrack=true"

    def get_id_list(self):
        settings = get_project_settings()
        self.infile_fname = settings.get("INFILE")
        saved_file_path = self.save_dir/f"id_counter.txt"
        ctr = 0 # default, read from the beginning of the id file
        if saved_file_path.exists():
            with open(saved_file_path, "r") as f:
                ctr = int(f.read())
        with open(self.infile_fname, "r") as f:
            for _ in range(ctr): # skip the lines that we have already fetched
                next(f)
        return [json.loads(line)['id'] for line in f]
    
    def start_requests(self):
        for idx in self.ids:
            yield scrapy.Request(
                url=self.base_url.format(idx), callback=self.parse
            )

    def parse(self, response):
        result = json.loads(response.text)
        yield result

    def closed(self, reason):
        settings = get_project_settings()
        self.output_fname = settings.get("OUTFILE")
        def get_ids_counter():
            ids = [json.loads(line)["id"] for line in open(self.output_fname,"r")]
            return len(ids)
        ctr = get_ids_counter()

        with open(self.save_dir / f"id_counter.txt", "w") as f:
            f.write(str(ctr))
