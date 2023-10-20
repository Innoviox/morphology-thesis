a = open("inputs.txt")
b = open("outputs.txt")

c = open("seqseq.txt", "w")

for x, y in zip(a.readlines(), b.readlines()):
    c.write(f"{x.strip()}\t{y.strip()}\n")

c.close()
