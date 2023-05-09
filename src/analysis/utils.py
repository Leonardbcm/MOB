from src.models.torch_wrapper import OBNWrapper
import pandas, os, datetime

def load_real_prices(country, n=None):
    base_folder = os.environ["MOB"]
    filename = f"{country}_real_prices.csv"
    path = os.path.join(base_folder, filename)
    df = pandas.read_csv(path)
    try:
        df.period_start_time = [datetime.datetime.strptime(d, "%Y-%m-%dT%H:00:00.0")
                                for d in df.period_start_time]
    except:
        df.period_start_time = [datetime.datetime.strptime(d, "%Y-%m-%dT%H:00:00")
                                for d in df.period_start_time]
        
    df.period_start_date = [d.date() for d in df.period_start_time]
    df.set_index("period_start_time", inplace=True, drop=True)
    
    if n is None:
        return df
    else:
        return df.head(n)
    
def get_model_wrappers(combinations, countries, datasets, spliter):
    model_wrappers = []
    for i, (skip_connection, use_order_book,  separate_optim, order_book_size) in enumerate(combinations):
        for j, (country, dataset) in enumerate(zip(countries, datasets)):
            model_wrapper = OBNWrapper(
                "TEST", dataset, spliter=spliter, country=country,
                skip_connection=skip_connection, use_order_books=use_order_book,
                order_book_size=order_book_size, separate_optim=separate_optim)
            model_wrappers += [model_wrapper]
    return model_wrappers

def handle_latex_string(header):
    splits = header.split("_")
    ss = ""
    for s in splits:
        ss += s + "\_"
    
    return ss[:-2]

def df_to_latex(df, index=True, index_col="", hlines=True, roundings=[],
                highlight=[]):    
    headers = [index_col] + list(df.columns)
    headers = [handle_latex_string(h) for h in headers]
    nc = len(headers)
    col_params = "|"
    tab_cols = ""
    for header in headers:
        tab_cols += "\\textbf{" + header + "} & "
        col_params += "c|"
    tab_cols = tab_cols[:-2]

    if highlight == []:
        highlight = ["" for header in headers]

    if roundings == []:
        roundings = [2 for header in headers]
    try:
        roundings[0]
    except:
        roundings = [roundings for header in headers]

    s = ""
    # DEFINE TABULAR 
    s += """
    \\begin{table}[htb]
    \\begin{center}
          \scalebox{1}{
            \\begin{tabular}{"""
    s += col_params
    s += """}
              \hline
    """
    s += tab_cols + "\\\\"
    s += """
              \hline
    """
    if hlines:
        s += """
                  \hline
        """

    rows = list(df.index)
    print(rows)
    nr = len(rows)
    for row in rows:
        # Add the index in first column
        i_row = str(row)        
        if index:
            i_row = "\\textbf{" + i_row + "}"
        s += i_row
        values = df.loc[row].values        
        for v, highlight_, col, rounding in zip(
                values, highlight, df.columns, roundings):
            str_v = str(round(v, ndigits=rounding))
            if str_v == "nan": str_v = " - "
            str_v_highlighted = "{\\bf "+ str_v + "}"
            if ((highlight_ == "high") and (v == max(df.loc[:, col]))) or ((highlight_ == "low") and (v == min(df.loc[:, col]))):
                str_v = str_v_highlighted
            s += " & " + str_v
        s += """\\\\
        """
        if hlines:
            s += """\hline
            """

    if not hlines: s+= """\hline
    """
    s +=  """ \end{tabular}  
    }
    \end{center}
    \caption{}
    \label{}
    \end{table}"""
    return s
