import pandas as pd


filename = 'df_info_train_miff'
df = pd.read_csv(filename + '.csv')
df_high = df.loc[df['PM2.5'] > 115]
df_low = df.loc[df['PM2.5'] <= 115]
low_sample, high_sample = len(df_low), len(df_high)
print("Many : Few, {} : {}, Ratio: {}".format(
    low_sample, high_sample, low_sample / high_sample)
)

ratio = [1, 3, 6, 9, 12, 15]
for r in ratio:
    low = df_low.sample(n=r*high_sample, random_state=1024)  # TODO: only for Beijing dataset
    print('New data split, low: {}, high: {}'.format(len(low), len(df_high)))
    df_Xys = pd.concat([low, df_high], ignore_index=True)
    df_Xys.to_csv('{}_ratio_{}.csv'.format(filename, r))