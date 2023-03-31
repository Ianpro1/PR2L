import pstats, cProfile
from speedtest import speedtest2

cProfile.runctx("speedtest2()", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()