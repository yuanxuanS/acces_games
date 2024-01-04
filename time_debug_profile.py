import pstats
p = pstats.Stats("prosvrp2.stats")
p.sort_stats("cumulative")  #["cumulative"]
p.print_stats()