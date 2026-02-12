
import weaviate
wc = weaviate.Client( url="http://localhost:8080")

from pprint import pprint
s=wc.schema.get()
for k in s.keys():
    print(f"= {k} =")
    if k == "classes":
        pprint(s[k])
