#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 22:11:58 2024

@author: glavrent
"""
import numpy as np
import pandas as pd

def combine_dataframes(dfs):
    """
    Combines a list of two-dimensional pandas DataFrames into a single DataFrame.
    Each original DataFrame is collapsed into a single row.
    The resulting columns are labeled as "row_label-column_label" from the original.
    
    Parameters
    ----------
    dfs : list of pd.DataFrame
        The input dataframes.
        
    Returns
    -------
    pd.DataFrame
        A combined DataFrame where each input DataFrame becomes a single row.
    """
    # List to hold the single-row DataFrames
    rows = []
    
    for df in dfs:
        # Stack the dataframe to get a MultiIndex Series (col_label, row_label) -> value
        stacked = df.stack()
        
        # Convert the MultiIndex series into a dictionary: "row-col" -> value
        row_dict = {f"{r}_{c}": val for (c, r), val in stacked.items()}
        
        # Create a single-row DataFrame from the dictionary
        single_row_df = pd.DataFrame([row_dict])
        
        # Append to list
        rows.append(single_row_df)
    
    # Concatenate all single-row DataFrames into one
    combined = pd.concat(rows, ignore_index=True)
    
    return combined


