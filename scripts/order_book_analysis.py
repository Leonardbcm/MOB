%load aimport

import matplotlib, numpy as np, os

from src.euphemia.orders import *
from src.euphemia.order_books import *
from src.euphemia.solvers import *
from src.euphemia.ploters import *

from src.analysis.utils import load_real_prices
from src.analysis.order_book_analysis_utils import *

base_folder = os.environ["MOB"]
data_folder = os.path.join(base_folder, "HOURLY")

##################### Compute price distribution
df = load_real_prices()
datetimes = df.index
date_time = datetimes[0]

OB = LoadedOrderBook(date_time, data_folder)
po_bins = np.arange(-500, 3000, 1)
pop_bins = np.arange(-500, 3000, 1)
p_bins = np.arange(-3500, 3500, 1)

po_values = np.histogram(OB.p0s, bins=po_bins)[0]
p_values = np.histogram(OB.prices, bins=p_bins)[0]
pop_values = np.histogram(OB.prices + OB.p0s, bins=pop_bins)[0]
for date_time in datetimes[1:]:
    OB = LoadedOrderBook(date_time, data_folder)

    po_values += np.histogram(OB.p0s, bins=po_bins)[0]
    p_values += np.histogram(OB.prices, bins=p_bins)[0]
    pop_values += np.histogram(OB.p0s + OB.prices, bins=pop_bins)[0]
    
    
####################### Plot results
with matplotlib.rc_context({ "text.usetex" : True,
                             "text.latex.preamble" : r"\usepackage[bitstream-charter]{mathdesign} \usepackage[T1]{fontenc} \usepackage{mathtools}",
                             "font.family" : ""}):
    plt.close("all")
    price_distribution(po_bins, po_values, p_bins, p_values, pop_bins, pop_values,
                       fontsize=20, plot_approx=False)
    plt.show()
    

100 * np.sum(p_values[np.argmax(p_values):np.argmax(p_values)+3]) / np.sum(p_values)
p_bins[np.argmax(p_values):np.argmax(p_values)+3]

##################### Compute the number pf FR, FA, PA
# Divide the OrderBook in rejected, accepted and partially accepted part
OB = LoadedOrderBook(datetimes[0], data_folder)
solver = TorchSolution(TorchOrderBook(OB.orders))
pstar = solver.solve().item()
fully_rejected, fully_accepted, partially_accepted = OB.divide_order_book(pstar)

# Solve compute the number of orders of each part.
solver.compute_ratios(pstar)

results = pandas.DataFrame(
    index=datetimes,
    columns=["fully_rejected", "fully_accepted", "partially_accepted"])
for date_time in datetimes:
    OB = LoadedOrderBook(date_time, data_folder)
    solver = TorchSolution(TorchOrderBook(OB.orders))
    pstar = solver.solve().item()
    fr, fa, pa = solver.compute_ratios(pstar)
    
    results.loc[date_time, "fully_rejected"] = fr
    results.loc[date_time, "fully_accepted"] = fa
    results.loc[date_time, "partially_accepted"] = pa

with matplotlib.rc_context({ "text.usetex" : True,
                             "text.latex.preamble" : r"\usepackage[bitstream-charter]{mathdesign} \usepackage[T1]{fontenc} \usepackage{mathtools}",
                             "font.family" : ""}):
    plt.close("all")
    number_of_orders(results)
    plt.show()
