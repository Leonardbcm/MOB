# MOB

Modeling Order Books.

## Installation

Our requirements where produced using `python3.8`

Install [torch](https://pytorch.org/get-started/locally/) (version 1.13):

> pip3 install torch --extra-index-url https://download.pytorch.org/whl/cpu

Install other requirements:

> pip install -r requirements.txt

Set up the `MOB` environment variable for reading/storing data:

> os.environ["MOB"] = path/to/folder

Install shap from 

> python -m pip install git+https://github.com/dsgibbons/shap.git

## Data

Data is available [here](https://www.dropbox.com/sh/z5m1udr3i7cl8sx/AAA2H5ErEqWl83nTIQ9Fzsdxa?dl=0).
This contains:

- 1 month of Orders Books history in `HOURLY`
- 6 year-length dataset of exogenous features in `data/datasets/Lyon`
- Experimental results 

## Scripts

The `script` folder contains the following:

- `example.py` to plot several Order Books (aggregated curves).
- `compare_methods.py` will solve the Dual Problem for several implemented solutions. Solutions with parameters (e.g. : `k` for sigmoid approximation) will test several values.
- `obn.py` fits an Order Book Network and inspect the forecasted order books along time.
- `order_book_analysis.py` extract information from the order book dataset : price distribution, acceptance distribution.



