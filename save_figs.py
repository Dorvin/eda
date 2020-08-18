import processing

datas = ["data0", "data1"]
under_score = "_"
invs = ["norm",]

print("Saving figs...")
for data in datas:
    for inv in invs:
        processing.get_fig(f"{data}{under_score}{inv}", 1500)
print("Done!")
