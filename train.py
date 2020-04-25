import os
from feature_extractor import load_extract

# directory with feature data
fname = "./features/"

if not os.path.exists(fname):
    os.mkdir(fname)

df_btc = load_extract('bitcoin') #loads bitcoin

df_eth = load_extract('ethereum') #loads ethereum

df_dash = load_extract('dash') #loads dash

df_ltc = load_extract('litecoin') #loads litecoin

df_xmr = load_extract('monero') #loads monero

df_xrp = load_extract('ripple') #loads ripple

# save dataframes with new features
df_btc.to_csv(fname + "bitcoin.csv")
df_eth.to_csv(fname + "ethereum.csv")
df_dash.to_csv(fname + "dash.csv")
df_ltc.to_csv(fname + "litecoin.csv")
df_xmr.to_csv(fname + "monero.csv")
df_xrp.to_csv(fname + "ripple.csv")
