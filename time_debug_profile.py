import pstats
p = pstats.Stats("svrp50_psro.stats")
p.sort_stats("cumulative")  #["cumulative"]
p.print_stats()