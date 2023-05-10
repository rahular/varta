# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem
from bs4 import BeautifulSoup as bs
import json

class DhScrapyPipeline:
	# cleansing HTML data
	def __init__(self,output_fname):
		self.output_fname = output_fname
		self.LOWER_LIMIT = 50

	@classmethod
	def from_crawler(cls, crawler):
		return cls(
			output_fname=crawler.settings.get("OUTFILE"),
		)

	def open_spider(self, spider):
		self.file = open(self.output_fname, 'a',encoding="utf-8")

	def close_spider(self, spider):

		def get_ids_counter():
			ids = [json.loads(line)["id"] for line in open(self.output_fname,"r")]
			return len(ids)
		
		ctr = get_ids_counter()
		
		with open(spider.save_dir / f"id_counter.txt", "w") as f:
			f.write(str(ctr))
		self.file.close()

	def process_item(self, item, spider):
		result = ItemAdapter(item)
		if result["morecontent"] and result["morecontent"].get("code", -100) == 200:
			result = result["morecontent"]["data"]
			try:
				soup = bs(
					result["content"] + result.get("content2", ""),
					"html.parser",
				)
				for elem in soup.find_all("blockquote"):
					elem.extract()
			except Exception:
				raise DropItem(f"Error processing {item}")
			else:
				word_count = 0
				if soup.text:
					word_count = len(soup.text.split())
				if (
					word_count > self.LOWER_LIMIT
					and "Continue reading this story on the publisher website"
					not in soup.text
				):
					obj = {
						"id": result["id"],
						"url": result.get("deeplinkUrl", ""),
						"headline": result.get("title", ""),
						"publication_date": result.get("publishTime", ""),
						"text": soup.text,
						"tags": set(),
						"reactions": {},
						"source_media": "",
						"source_url": result.get("publisherStoryUrl", ""),
						"word_count": word_count,
						"langCode": result.get("langCode", ""),
					}
					if result.get("source", None):
						obj["source_media"] = result["source"].get("nameEnglish", "")
					if result.get("categoryKey", None):
						obj["tags"].add(result["categoryKey"])
					if result.get("hashtags", None):
						obj["tags"].update(
							[tag["name"][1:] for tag in result["hashtags"]]
						)
					if result.get("counts", None):
						for reaction in result["counts"]:
							if reaction != "TOTAL_LIKE":
								obj["reactions"][reaction] = result["counts"][reaction][
									"value"
								]
					obj["tags"] = list(obj["tags"])
					line = json.dumps(ItemAdapter(obj).asdict(),ensure_ascii=False) + "\n"
					self.file.write(line)
					return obj
				else:
					raise DropItem(f"Article too short {item}")
		else:
			raise DropItem(f"Error processing {item}")

	
