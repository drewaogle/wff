
import weaviate
wc = weaviate.Client( url="http://localhost:8080")

#https://weaviate-python-client.readthedocs.io/en/v3.2.3/weaviate.data.html

#col = wc.collections.use("Embeddings_vggface_retinaface_aligned_raw")
first=True
to_d=[]
for itm in wc.data_object.get()['objects']:
    #print(itm.uuid, itm.properties)
    if first:
        first=False
        print( "Props: {}".format( ",".join( itm['properties'].keys())))
    print(f"* {itm['class']} {itm['id']}")
    to_d.append(itm['id'])
    #"objects" ?
for d in to_d:
    r=wc.data_object.delete(d)
    print(r)
