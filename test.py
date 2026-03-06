from pprint import pprint

from dune_client.client import DuneClient
dune = DuneClient("w8VOl9Xry9yhDIfOUIGhN45iiVgPwtlq")
query_result = dune.get_latest_result(6765135)
pprint(query_result)