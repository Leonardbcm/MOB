import pandas, os, datetime

def load_real_prices(n=None):
    base_folder = os.environ["MOB"]
    filename = "real_prices.csv"
    path = os.path.join(base_folder, filename)
    df = pandas.read_csv(path)
    df.period_start_time = [datetime.datetime.strptime(d, "%Y-%m-%dT%H:00:00")
                            for d in df.period_start_time]
    df.period_start_date = [d.date() for d in df.period_start_time]
    df.set_index("period_start_time", inplace=True, drop=True)
    
    if n is None:
        return df
    else:
        return df.head(n)
