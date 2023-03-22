%load aimport

import matplotlib, torch

print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total", row_limit=30))
