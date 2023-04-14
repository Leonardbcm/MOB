import datetime, numpy as np, pandas

def OB_to_csv(OB, datetimes):
    """
    Converts a numpy OB to a dataframe and save it
    """
    OBs = OB.shape[1]
    variables = ["V", "Po", "P"]
    cols = [f"OB_{h}_{v}_{ob}" for h in range(24)
          for v in variables for ob in range(OBs)]
    past_cols = [f"OB_{h}_{v}_{ob}_past_1" for h in range(24)
                 for v in variables for ob in range(OBs)]
    
    # daylight saving time
    days_to_add = [datetime.date(2016, 3, 27),
                   datetime.date(2017, 3, 26),
                   datetime.date(2018, 3, 25),
                   datetime.date(2019, 3, 31),
                   datetime.date(2020, 3, 29),
                   datetime.date(2021, 3, 28)]
    dates = [datetime.date(d.year, d.month, d.day) for d in datetimes]
    
    u_dates = list(np.unique(dates))

    res = pandas.DataFrame(index = u_dates, columns=past_cols + cols)
    for i, (dt, d) in enumerate(zip(datetimes, dates)):
        h = dt.hour
        d_past = d + datetime.timedelta(hours=24)
        
        for j, v in enumerate(variables):
            data = OB[i, :, j].reshape(-1)
            
            cs = [f"OB_{h}_{v}_{ob}" for ob in range(OBs)]
            res.loc[d, cs] = data.copy()
            
            # Fill past
            cs_past = [f"OB_{h}_{v}_{ob}_past_1" for ob in range(OBs)]  
            if d_past <= dates[-1]:                        
                res.loc[d_past, cs_past] = data.copy()

            # Also fill daylight saving times
            if (d in days_to_add) and (h == 1):
                csnext = [f"OB_2_{v}_{ob}" for ob in range(OBs)]                
                res.loc[d, csnext] = data.copy()

                cs_past = [f"OB_2_{v}_{ob}_past_1" for ob in range(OBs)]  
                if d_past <= dates[-1]:                        
                    res.loc[d_past, cs_past] = data.copy()
                            
    res.sort_index(inplace=True)
    return res
