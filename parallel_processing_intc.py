
import multiprocessing
from hybrid_hawkes_exp import HybridHawkesExp

import pandas as pd
import numpy as np

intc_messages = pd.read_csv("INTC_data/INTC_2023-09-01_34200000_57600000_message_10.csv")
intc_lob = pd.read_csv("INTC_data/INTC_2023-09-01_34200000_57600000_orderbook_10-Copy1.csv")
intc = pd.concat((intc_messages, intc_lob), axis = 1)

intc_messages5 = pd.read_csv("INTC_data/INTC_2023-09-05_34200000_57600000_message_10.csv")
intc_lob5 = pd.read_csv("INTC_data/INTCcopy_2023-09-05_34200000_57600000_orderbook_10.csv")
intc5 = pd.concat((intc_messages5, intc_lob5), axis = 1)

intc_messages6 = pd.read_csv("INTC_data/INTC_2023-09-06_34200000_57600000_message_10.csv")
intc_lob6 = pd.read_csv("INTC_data/INTCcopy_2023-09-06_34200000_57600000_orderbook_10.csv")
intc6 = pd.concat((intc_messages6, intc_lob6), axis = 1)

intc_messages7 = pd.read_csv("INTC_data/INTC_2023-09-07_34200000_57600000_message_10.csv")
intc_lob7 = pd.read_csv("INTC_data/INTCcopy_2023-09-07_34200000_57600000_orderbook_10.csv")
intc7 = pd.concat((intc_messages7, intc_lob7), axis = 1)

intc_messages8 = pd.read_csv("INTC_data/INTC_copy2023-09-08_34200000_57600000_message_10.csv")
intc_lob8 = pd.read_csv("INTC_data/INTCcopy_2023-09-08_34200000_57600000_orderbook_10.csv")
intc8 = pd.concat((intc_messages8, intc_lob8), axis = 1)

intc_messages11= pd.read_csv("INTC_data/INTCcopy_2023-09-11_34200000_57600000_message_10.csv")
intc_lob11 = pd.read_csv("INTC_data/INTCcopy_2023-09-11_34200000_57600000_orderbook_10.csv")
intc11 = pd.concat((intc_messages11, intc_lob11), axis = 1)

intc_messages12= pd.read_csv("INTC_data/INTCcopy_2023-09-12_34200000_57600000_message_10.csv")
intc_lob12 = pd.read_csv("INTC_data/INTCcopy_2023-09-12_34200000_57600000_orderbook_10.csv")
intc12 = pd.concat((intc_messages12, intc_lob12), axis = 1)

intc_messages13= pd.read_csv("INTC_data/INTCcopy_2023-09-13_34200000_57600000_message_10.csv")
intc_lob13 = pd.read_csv("INTC_data/INTCcopy_2023-09-13_34200000_57600000_orderbook_10.csv")
intc13 = pd.concat((intc_messages13, intc_lob13), axis = 1)

intc_messages14= pd.read_csv("INTC_data/INTCcopy_2023-09-14_34200000_57600000_message_10.csv")
intc_lob14 = pd.read_csv("INTC_data/INTCcopy_2023-09-14_34200000_57600000_orderbook_10.csv")
intc14 = pd.concat((intc_messages14, intc_lob14), axis = 1)

intc_messages15= pd.read_csv("INTC_data/INTCcopy_2023-09-15_34200000_57600000_message_10.csv")
intc_lob15 = pd.read_csv("INTC_data/INTCcopy_2023-09-15_34200000_57600000_orderbook_10.csv")
intc15 = pd.concat((intc_messages15, intc_lob15), axis = 1)

intc_messages18= pd.read_csv("INTC_data/INTCcopy_2023-09-18_34200000_57600000_message_10.csv")
intc_lob18 = pd.read_csv("INTC_data/INTCcopy_2023-09-18_34200000_57600000_orderbook_10.csv")
intc118 = pd.concat((intc_messages18, intc_lob18), axis = 1)

intc_messages19= pd.read_csv("INTC_data/INTCcopy_2023-09-19_34200000_57600000_message_10.csv")
intc_lob19 = pd.read_csv("INTC_data/INTCcopy_2023-09-19_34200000_57600000_orderbook_10.csv")
intc19 = pd.concat((intc_messages19, intc_lob19), axis = 1)

intc_messages22= pd.read_csv("INTC_data/INTCCopy_2023-09-22_34200000_57600000_orderbook_10.csv")
intc_lob22= pd.read_csv("INTC_data/INTCCopy_2023-09-22_34200000_57600000_orderbook_10.csv")
intc122 = pd.concat((intc_messages22, intc_lob22), axis = 1)

intc_messages21= pd.read_csv("INTC_data/INTCCopy_2023-09-22_34200000_57600000_message_10.csv")
intc_lob21= pd.read_csv("INTC_data/INTCCopy_2023-09-22_34200000_57600000_orderbook_10.csv")
intc22 = pd.concat((intc_messages22, intc_lob22), axis = 1)

import pandas as pd

# Define the list of dates to iterate over
dates = ["2023-09-11", "2023-09-12", "2023-09-13", "2023-09-14", "2023-09-15",
         "2023-09-18", "2023-09-19", "2023-09-20", "2023-09-21", "2023-09-22",
         "2023-09-25", "2023-09-26", "2023-09-27", "2023-09-28", "2023-09-29"]

# Create a dictionary to store the dataframes
intc_dataframes = {}

# Iterate over each date
for date in dates:
    # Read the message and orderbook CSV files for the current date
    messages_file = f"INTC_data/INTCcopy_{date}_34200000_57600000_message_10.csv"
    lob_file = f"INTC_data/INTCcopy_{date}_34200000_57600000_orderbook_10.csv"
    
    # Read CSV files
    messages_df = pd.read_csv(messages_file)
    lob_df = pd.read_csv(lob_file)
    
    # Concatenate the dataframes horizontally
    concatenated_df = pd.concat((messages_df, lob_df), axis=1)
    
    # Store the concatenated dataframe in the dictionary with key as date
    intc_dataframes[f'intc_{date.replace("-", "_")}'] = concatenated_df

    intc11 = intc_dataframes['intc_2023_09_11']
intc12 = intc_dataframes['intc_2023_09_12']
intc13 = intc_dataframes['intc_2023_09_13']
intc14 = intc_dataframes['intc_2023_09_14']
intc15 = intc_dataframes['intc_2023_09_15']
intc18 = intc_dataframes['intc_2023_09_18']
intc19 = intc_dataframes['intc_2023_09_19']
intc20 = intc_dataframes['intc_2023_09_20']
intc21 = intc_dataframes['intc_2023_09_21']
intc22 = intc_dataframes['intc_2023_09_22']
intc25 = intc_dataframes['intc_2023_09_25']
intc26 = intc_dataframes['intc_2023_09_26']
intc27 = intc_dataframes['intc_2023_09_27']
intc28 = intc_dataframes['intc_2023_09_28']
intc29 = intc_dataframes['intc_2023_09_29']

intc_earlier_df = {"intc_2023_09_05":intc5,"intc_2023_09_06":intc6,"intc_2023_09_07":intc7, "intc_2023_09_08":intc8}

intc_earlier_df.update(intc_dataframes)

# we group trades based on events that a) add liquidity and b) take liquidity 
def group_trades(data, type_column, direction_column):
    # buy, liquidity taking - buy, limit orders arrive/ executed
    data1 = data[(data[type_column]==1) & (data[direction_column])==1]
    data2 = data[(data[type_column]==4) & (data[direction_column])==1]
    data3 = data[(data[type_column]==5) & (data[direction_column])==1]
    # sell cancellations - liquidity taking 
    data4 = data[(data[type_column]==2) & (data[direction_column])==-1]
    data5 = data[(data[type_column]==3) & (data[direction_column])==-1]
    data_lt = pd.concat((data1, data2, data3, data4, data5), axis =0)
    data_lt = data_lt.sort_values(by = "time")
    # liquidity adding - buy orders cancelled
    data6 = data[(data[type_column]==2) & (data[direction_column])==1]
    data7 = data[(data[type_column]==3) & (data[direction_column])==1]
    # liquidity adding - sell orders received
    data8 = data[(data[type_column]==1) & (data[direction_column])==-1]
    data9 = data[(data[type_column]==4) & (data[direction_column])==-1]
    data10 = data[(data[type_column]==5) & (data[direction_column])==-1]
    data_la = pd.concat((data6, data7, data8, data9, data10), axis =0)
    data_la = data_la.sort_values(by = "time")
    return data_lt, data_la

def classify_trades(data, type_column, direction_column):
    """classify - state 1 = liquidity adding, state 2 = liquidity taking"""
    # buy, liquidity taking - buy, limit orders arrive/ executed
    data.loc[(data[type_column]==1) & (data[direction_column]==1), 'event'] = 0
    data.loc[(data[type_column]==4) & (data[direction_column]==1), 'event'] = 0
    data.loc[(data[type_column]==5) & (data[direction_column]==1), 'event'] = 0
    # sell cancellations - liquidity taking 
    data.loc[(data[type_column]==2) & (data[direction_column]==-1), 'event'] = 0
    data.loc[(data[type_column]==3) & (data[direction_column]==-1), 'event'] = 0
    # liquidity adding - buy orders cancelled
    data.loc[(data[type_column]==2) & (data[direction_column]==1), 'event'] = 1
    data.loc[(data[type_column]==3) & (data[direction_column]==1), 'event'] = 1
    # liquidity adding - sell orders received
    data.loc[(data[type_column]==1) & (data[direction_column]==-1), 'event'] = 1
    data.loc[(data[type_column]==4) & (data[direction_column]==-1), 'event'] = 1
    data.loc[(data[type_column]==5) & (data[direction_column]==-1), 'event'] = 1
    return data

def classify_bid_ask(data, bid_col, ask_col):
    ## calculate spread, and classify as 1 tick or more (2-state classification)
    data['spread'] = data[ask_col]-data[bid_col]
    data['state'] = 0
    data.loc[data['spread']==100, 'state']=0
    data.loc[data['spread']>100, 'state']=1
    return data

def time_extract(data, time_column, time1, time2):
    """time1, time2 should be time-representing strings"""
    data_time = data[(pd.to_datetime(data['time'], unit = "s").dt.time > pd.to_datetime(time1).time())&(pd.to_datetime(data['time'], unit = "s").dt.time < pd.to_datetime(time2).time())]
    return data_time 

intc5 = classify_trades(intc5, "type", "direction")
intc5 = classify_bid_ask(intc5, "bid_level1", "ask_level1")

intc6 = classify_trades(intc6, "type", "direction")
intc6 = classify_bid_ask(intc6, "bid_level1", "ask_level1")

intc7 = classify_trades(intc7, "type", "direction")
intc7 = classify_bid_ask(intc7, "bid_level1", "ask_level1")

intc8 = classify_trades(intc8, "type", "direction")
intc8 = classify_bid_ask(intc8, "bid_level1", "ask_level1")

intc11 = classify_trades(intc11, "type", "direction")
intc11 = classify_bid_ask(intc11, "bid_level1", "ask_level1")

intc12 = classify_trades(intc12, "type", "direction")
intc12 = classify_bid_ask(intc12, "bid_level1", "ask_level1")

intc13 = classify_trades(intc13, "type", "direction")
intc13 = classify_bid_ask(intc13, "bid_level1", "ask_level1")

intc14 = classify_trades(intc14, "type", "direction")
intc14 = classify_bid_ask(intc14, "bid_level1", "ask_level1")

intc15 = classify_trades(intc15, "type", "direction")
intc15 = classify_bid_ask(intc15, "bid_level1", "ask_level1")

intc118 = classify_trades(intc18, "type", "direction")
intc18 = classify_bid_ask(intc18, "bid_level1", "ask_level1")

intc19= classify_trades(intc19, "type", "direction")
intc19 = classify_bid_ask(intc19, "bid_level1", "ask_level1")

intc20= classify_trades(intc20, "type", "direction")
intc20 = classify_bid_ask(intc20, "bid_level1", "ask_level1")

intc21= classify_trades(intc21, "type", "direction")
intc21 = classify_bid_ask(intc21, "bid_level1", "ask_level1")

intc22= classify_trades(intc22, "type", "direction")
intc22 = classify_bid_ask(intc22, "bid_level1", "ask_level1")

intc25= classify_trades(intc25, "type", "direction")
intc25 = classify_bid_ask(intc25, "bid_level1", "ask_level1")

intc26= classify_trades(intc26, "type", "direction")
intc26 = classify_bid_ask(intc26, "bid_level1", "ask_level1")

intc27= classify_trades(intc27, "type", "direction")
intc27 = classify_bid_ask(intc27, "bid_level1", "ask_level1")
# skip 28th september because trading is halted early

intc29= classify_trades(intc29, "type", "direction")
intc29 = classify_bid_ask(intc29, "bid_level1", "ask_level1")


alphas_hat_dict = {}
betas_hat_dict = {}
base_rates1_dict = {}
base_rates2_dict = {}

intc_data = {'intc5': intc5, 'intc6': intc6, 'intc7': intc7, 'intc8': intc8, 'intc11': 
             intc11, 'intc12': intc12, 'intc13': intc13, 'intc14': intc14, 'intc15': intc15, 
             'intc18': intc18, 'intc19': intc19, 'intc20': intc20, 'intc21': intc21, 'intc22': intc22, 
             'intc25': intc25, 'intc26': intc26, 'intc27': intc27, 'intc29': intc29}

def base_rates_trading_day(time_string, time_vector, data, time_col, event_col, state_col, alphas_hat, betas_hat):
    base_rates_1 = []
    base_rates_2 = []
    for i in range(len(time_string)-1):
        data_interval = time_extract(data, time_col, time_string[i], time_string[i+1])
        times_interval = data_interval[time_col].to_numpy()
        events_interval = data_interval[event_col].to_numpy().astype(int)
        states_interval = data_interval[state_col].to_numpy().astype(int)
        opt_result, initial_guess, initial_guess_kind = model.estimate_hawkes_base_rate(times_interval, events_interval, states_interval,
                                                                                 time_vector[i], time_vector[i+1], alphas_hat, betas_hat,)
        base_rate_1 = opt_result.x[0]
        base_rate_2 = opt_result.x[1]
        base_rates_1.append(base_rate_1)
        base_rates_2.append(base_rate_2)
    base_rates_1.append(base_rate_1)
    base_rates_2.append(base_rate_2)

    return base_rates_1, base_rates_2

import multiprocessing
import numpy as np
import matplotlib.pyplot as plt

# Function to process each DataFrame
def process_dataframe(df_key, df_full):
    opt_result, initial_guess, initial_guess_kind = model.estimate_hawkes_parameters(
        df_full['time'].to_numpy(), 
        df_full['event'].astype(int).to_numpy(), 
        df_full['state'].to_numpy(), 
        34200, 
        57600
    )
    # Extract the MLE estimate
    mle_estimate = opt_result.x
    nus_hat, alphas_hat, betas_hat = model.array_to_parameters(mle_estimate, n_events, n_states)
    
    # Compute base rates for the trading day
    time_strings = ['9:30', '10:00', "10:30", "11:00", "11:30", "12:00", "12:30", "13:00", "13:30", "14:00", "14:30", "15:00", "15:30", "16:00"]
    base_rates_1 = base_rates_trading_day(time_strings, list(np.array([1800 * n for n in range(14)])+34200), df_full, 
                                          "time","event", "state", alphas_hat, betas_hat)[0]
    base_rates_2 = base_rates_trading_day(time_strings, list(np.array([1800 * n for n in range(14)])+34200), df_full,
                                          "time","event", "state", alphas_hat, betas_hat)[1]
    return df_key, base_rates_1, base_rates_2

def plot_base_rate(time_strings, base_rates_1, base_rates_2, date_string, time_interval_string):
    # Plot piece-wise constant functions
    plt.figure(figsize=(18, 7))
    plt.step(time_strings, base_rates_1, where='post', label='Event Type 1')
    plt.step(time_strings, base_rates_2, where='post', label='Event Type 2')
    plt.xlabel('Time')
    plt.ylabel('Base Rate')
    plt.title(f'Piece-wise Constant Intraday Base Rates, {time_interval_string} intervals,\nDate: {date_string}')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close()

if __name__ == "__main__":
    # Create a pool of processes
    pool = multiprocessing.Pool()

    # Iterate over the dictionary items
    results = []
    for df_key, df_full in intc_data.items():
        # Apply multiprocessing to each DataFrame
        result = pool.apply_async(process_dataframe, (df_key, df_full))
        results.append(result)

    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()

    # Retrieve the results
    for result in results:
        df_key, base_rates_1, base_rates_2 = result.get()
        alphas_hat_dict[df_key] = alphas_hat
        betas_hat_dict[df_key] = betas_hat
        base_rates1_dict[df_key] = base_rates_1
        base_rates2_dict[df_key] = base_rates_2

    # Plot base rates for a specific date
    plot_base_rate(time_strings, base_rates1_dict["intc5"], base_rates2_dict['intc5'], "INTC, 5th Sept 2023", "30 mins")
