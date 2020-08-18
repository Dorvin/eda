import urllib.request
#import requests
#import wget

def download_csv(tag = "loss", dataset = "data1", inv = "norm", tv = "train", depth = "d0"):
  source = f"http://localhost:6006/data/plugin/scalars/scalars?tag={tag}&run={dataset}_{inv}%2F{tv}%2F{depth}&experiment=&format=csv"
  target = f"./result/{dataset}_{inv}/{tv}/{depth}_{tag}.csv"
  try:
    urllib.request.urlretrieve(source, target)
  except Exception as e:
    print("Failed: " + source + "   =>   " + target)
    print(e)

#def download_csv2(tag = "loss", dataset = "data1", inv = "norm", tv = "train", depth = "d0"):
#  source = f"http://localhost:6006/data/plugin/scalars/scalars?tag={tag}&run={dataset}_{inv}%2F{tv}%2F{depth}&experiment=&format=csv"
#  target = f"./result/{dataset}_{inv}/{tv}/{depth}_{tag}.csv"
#  try:
#    r = requests.get(source)
#    with open(target, 'wb') as f:
#      f.write(r.content)
#  except Exception as e:
#    print("Failed: " + source + "   =>   " + target)
#    print(e)

#def download_csv3(tag = "loss", dataset = "data1", inv = "norm", tv = "train", depth = "d0"):
#  source = f"http://localhost:6006/data/plugin/scalars/scalars?tag={tag}&run={dataset}_{inv}%2F{tv}%2F{depth}&experiment=&format=csv"
#  target = f"./result/{dataset}_{inv}/{tv}/{depth}_{tag}.csv"
#  try:
#    wget.download(source, target)
#  except Exception as e:
#    print("Failed: " + source + "   =>   " + target)
#    print(e)

tags = ["loss", "accuracy"]
datasets = ["data0", "data1"]
invs = ["norm",]
tvs = ["train", "val"]
depths = ["d0", "d1", "d5"]

print("Downloading csvs ...")
for tag in tags:
  for dataset in datasets:
    for inv in invs:
      for tv in tvs:
        for depth in depths:
          download_csv(tag = tag, dataset = dataset, inv = inv, tv = tv, depth = depth)
print("Done!")
