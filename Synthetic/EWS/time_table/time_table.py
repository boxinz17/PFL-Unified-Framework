import pandas as pd
import numpy as np
import pickle

optimizer_list = ['LSGD', 'ASCD', 'ASVRCD']
hetero_lv_list = [5.0, 10.0, 15.0]

time_list = []
for optim in optimizer_list:
    time_list2 = []
    for hetero_lv in hetero_lv_list:
        file_name = 'EWS_' + optim + '_time_lv' + str(hetero_lv) + '.txt'
        result_path = '../result/' + file_name
        with open(result_path, "rb") as f:
            time_used = pickle.load(f)
            time_used = np.array(time_used)
            time_list2.append(time_used.mean())
    time_list.append(time_list2)

df = pd.DataFrame(time_list, index=optimizer_list, columns=hetero_lv_list)
df = df.applymap('{:,.2f}'.format)
df.to_latex("latex_code.txt")