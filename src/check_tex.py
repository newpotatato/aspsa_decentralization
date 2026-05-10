text = open("c:/tarasova/chenikov_stars/paper/sections/experiments.tex", encoding="utf-8").read()
opens = text.count("{")
closes = text.count("}")
print("open=%d close=%d diff=%d" % (opens, closes, opens - closes))
for env in ["tabular", "table", "figure", "align", "equation"]:
    b = text.count("begin{" + env + "}")
    e = text.count("end{" + env + "}")
    print("  %s: begin=%d end=%d ok=%s" % (env, b, e, b == e))
