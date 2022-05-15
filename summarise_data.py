import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

Total_Volume = 3.0
GSH_concentration = 1.0
epsilon = 9.6
Cache_Length = 1.0

wrong_format_list = []


def model(t, K, S):
    return (1 - np.exp(-K * t)) * S * epsilon * Cache_Length


def fit_data(t, Abs):
    guess = [0.1, GSH_concentration]
    bounds = ([0, 0], [10, GSH_concentration])
    popt, pcov = curve_fit(
        f=model,
        xdata=t,
        ydata=Abs,
        p0=guess,
        bounds=bounds,
    )
    return popt


weekday_zh2en = {
    "周一": "Mon",
    "周二": "Tue",
    "周三": "Wed",
    "周四": "Thu",
    "周五": "Fri",
    "周六": "Sat",
    "周日": "Sun",
}

dataList = {}
for fileName in os.listdir("data"):
    df = pd.read_csv(os.path.join("data", fileName))
    if len(df) < 9:
        wrong_format_list.append(fileName)
        continue
    dataList[fileName[:-4]] = df
print("wrong format:", wrong_format_list)
# dataList = {
#     fileName[:-4]: pd.read_csv(f"./data/{fileName}") for fileName in os.listdir("data")
# }
df = pd.DataFrame(
    {
        "Method": [],
        "Enzyme_Activity_tot": [],
        "Enzyme_Activity_avg": [],
        "S_estimate": [],
        "R_squared": [],
        "Method_Group": [],
        "Day": [],
        "Group": [],
        "Name": [],
        "Id": [],
    }
)
for label, data in dataList.items():
    Day, Group, Name, Id = label.split("-")
    t = data["Minute"].values
    for colName in data.columns[1:]:
        Abs = data[colName].values
        Abs[0] = 0
        popt = fit_data(t, Abs)
        fit = model(t, *popt)
        K_estimate = popt[0]
        S_estimate = popt[1]
        K_tot = (1 - np.exp(-K_estimate)) * S_estimate * 3.0
        K_avg = (1 - np.exp(-K_estimate)) * S_estimate * 3.0 / 3.0
        R_squared = np.corrcoef(Abs, fit)[0, 1] ** 2
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "Method": colName,
                        "Enzyme_Activity_tot": K_tot,
                        "Enzyme_Activity_avg": K_avg,
                        "S_estimate": S_estimate,
                        "R_squared": R_squared,
                        "Method_Group": int(colName[:2] == "pH"),
                        "Day": weekday_zh2en[Day],
                        "Group": Group,
                        "Name": hex(hash(Name) + hash(Id))[-8:],
                        "Id": Id,
                    },
                    index=[0],
                ),
            ],
            ignore_index=True,
        )

df_dedup = df.drop_duplicates(subset=["Enzyme_Activity_tot", "Day", "Group", "Method"])
print("Before dedup")
print(df.shape)
print("After dedup")
print(df_dedup.shape)

group_means = df_dedup.groupby(["Day", "Group"]).mean()["Enzyme_Activity_avg"]
mean_means = group_means.mean()
norm_params = group_means / mean_means
# normalize data based on group
df_new_column = df_dedup[["Enzyme_Activity_avg", "Day", "Group"]].copy()
for idx, params in norm_params.items():
    day, group = idx
    df_new_column.loc[
        (df_new_column["Group"] == group) & (df_new_column["Day"] == day),
        "Enzyme_Activity_avg",
    ] /= params
# add a column of noramlized data to df_dedup
df_norm = df_dedup.copy()
df_norm.insert(
    loc=list(df_dedup.columns).index("Enzyme_Activity_avg") + 1,
    column="Enzyme_Activity_avg_norm",
    value=df_new_column["Enzyme_Activity_avg"],
)

# Save dataframes to csv
df_norm.to_csv("./result-Total.csv", index=False)
for day in df_norm["Day"].unique():
    df_day = df_norm[df_norm["Day"] == day]
    print(day)
    print(df_day.shape)
    df_day.to_csv(f"./result-{day}.csv", index=False)
