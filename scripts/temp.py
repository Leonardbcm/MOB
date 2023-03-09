################### Compare 2 order books with 24h interval
date = df.period_start_date[100]
hour = 7
date_time = datetime.datetime(date.year, date.month, date.day, hour)

OB1 = LoadedOrderBook(date_time, data_folder)
torch_book1 = TorchOrderBook(OB1.orders)

OB2 = LoadedOrderBook(date_time + datetime.timedelta(hours=24), data_folder)
torch_book2 = TorchOrderBook(OB2.orders)

torch_book1.n, torch_book2.n
all_p0s = set(torch_book1.p0s).union(set(torch_book2.p0s))
N = len(all_p0s)

common_p0s = set(torch_book1.p0s).intersection(set(torch_book2.p0s))

indices_tob1 = []
indices_tob2 = []
for p0 in common_p0s:
    indices_tob1 += [np.where(torch_book1.p0s == p0)[0][0]]
    indices_tob2 += [np.where(torch_book2.p0s == p0)[0][0]]

PBv = np.zeros(N)
PBpo = np.zeros(N)

##################### Compute price distribution
date_time = datetimes[0]

OB = LoadedOrderBook(date_time, data_folder)
po_bins = np.arange(-500, 3000, 1)
p_bins = np.arange(-3500, 3500, 1)

po_values = np.histogram(OB.p0s, bins=po_bins)[0]
p_values = np.histogram(OB.prices, bins=p_bins)[0]
for date_time in datetimes[1:]:
    OB = LoadedOrderBook(date_time, data_folder)

    po_values += np.histogram(OB.p0s, bins=po_bins)[0]
    p_values += np.histogram(OB.prices, bins=p_bins)[0]    

    
plt.bar(po_bins[1:], po_values, width=po_bins[1] - po_bins[0], color="m",
        label="Po", alpha=0.6)
plt.bar(p_bins[1:], p_values, width=p_bins[1] - p_bins[0],  color="c",
        label="P", alpha=0.6)
plt.legend()
plt.show()
    
plt.plot(po_bins[1:], np.cumsum(po_values), color="m", label="Po")
plt.plot(p_bins[1:], np.cumsum(p_values), color="c", label="P")
plt.legend()
plt.show()

100 * np.sum(p_values[np.argmax(p_values):np.argmax(p_values)+3]) / np.sum(p_values)
p_bins[np.argmax(p_values):np.argmax(p_values)+3]

##################### Compute the number pf FR, FA, PA
filename = "interpolated_price.csv"
path = os.path.join(base_folder, filename)
df = pandas.read_csv(path)
df.period_start_time = [datetime.datetime.strptime(d, "%Y-%m-%dT%H:00:00.0")
                        for d in df.period_start_time]
df.period_start_date = [d.date() for d in df.period_start_time]
df.set_index("period_start_time", inplace=True, drop=True)
datetimes = df.index



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
    params = {"fontsize" : 25, "label_fontsize" : 30, "linewidth" : 4}    
    plt.close("all")
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(19.2,10.8)
                                   ,gridspec_kw={"wspace" : 0.0})
    colors = ("r", "b", "g")
    for column, color in zip(results.columns, colors):    
        ax1.hist(results.loc[:, column].values, bins=100, color=color,
                 edgecolor="k",
                 alpha=0.6, label=column) 
        ax2.hist(100 * results.loc[:, column].values / results.values.sum(axis=1),
                 bins=100,color=color,edgecolor="k",alpha=0.6,label=column)

    ax1.hist(results.values.sum(axis=1), bins=100, color="y",
             edgecolor="k", alpha=0.3, label="Total") 
    
    ax1.set_title("Number of orders", fontsize=params["fontsize"])
    ax2.set_title("\% of orders", fontsize=params["fontsize"])
    ax1.grid("on")
    ax2.grid("on")

    ax1.tick_params(axis="both", labelsize=params["fontsize"])
    ax2.tick_params(axis="both", labelsize=params["fontsize"])    
    ax1.legend(fontsize=params["fontsize"])
    plt.show()
