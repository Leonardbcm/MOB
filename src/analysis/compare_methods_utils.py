import matplotlib.pyplot as plt, numpy as np, os, pandas

def it_results(fontsize=20):
    fig, [axk, axn] = plt.subplots(1, 2, figsize=(19.2, 10.8),
                                   sharex=False, sharey=True,
                           gridspec_kw={"wspace" : 0.0})

    base_folder = os.environ["MOB"]
    
    filename = "k_results.csv"
    path = os.path.join(base_folder, filename)
    k_results = pandas.read_csv(path)
    k_values = [int(col.split("generic_sigmoid_")[1])
                for col in k_results.columns if "generic_sigmoid" in col]
    k_maes = [k_results.loc[:, f"dual_derivative_generic_sigmoid_{k}"].values[0]
              for k in k_values]
    
    barwidth = 0.8    
    axk.bar(np.log10(k_values), k_maes, width=barwidth,
            edgecolor="k", color="b", alpha=0.6)
    axk.bar(
        np.max(np.log10(k_values)) + 2,
        k_results.loc[:, "dual_derivative_heaviside"].values[0],
        width=barwidth,
        edgecolor="k", color="r", alpha=0.6)    

    axk.grid("on", axis="y")
    axn.grid("on", axis="y")    
    axk.set_xticks([np.log10(i) for i in k_values] + [len(k_values) + 1])
    axk.set_xticklabels([f"k={k}" for k in k_values] + ["H"],
                        fontsize=fontsize)
    
    axk.set_ylabel("Reconstruction Error \euro{}/MWh", fontsize=fontsize)
    axk.set_xlabel("$\sigma(kx)$", fontsize=fontsize)

    filename = "niter_results.csv"
    path = os.path.join(base_folder, filename)
    niter_results = pandas.read_csv(path)
    niter_values = [int(col.split("BatchSolver")[1])
                    for col in niter_results.columns if "BatchSolver" in col]
    niter_maes = [niter_results.loc[:, f"BatchSolver{niter}"].values[0]
                  for niter in niter_values]
    
    barwidth = 4    
    axn.bar(niter_values, niter_maes, width=barwidth,
            edgecolor="k", color="g", alpha=0.6)
    
    axn.set_xticks(niter_values)
    axn.set_xticklabels([str(n) for n in niter_values], fontsize=fontsize)
    axn.set_xlabel("Number of iterations", fontsize=fontsize)

    miny_ = np.min(k_maes + niter_maes)
    maxy_ = np.max(k_maes)
    delta = 10 * (maxy_ - miny_) / 100
    miny = miny_ - delta
    maxy = maxy_ + delta
    axk.set_ylim([miny, maxy])    

    axk.set_title("Replacing Heaviside by Sigmoid", fontsize=fontsize)    
    axn.set_title("Batch Solving, k=100", fontsize=fontsize)
