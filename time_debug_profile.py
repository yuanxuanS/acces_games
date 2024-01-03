import pstats
p = pstats.Stats("profile_act_no_contebd.stats")
p.sort_stats("cumulative")  #["cumulative"]
p.print_stats()