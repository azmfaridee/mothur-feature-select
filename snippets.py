# temp = pearsons[pearsons < corr_threshold]
# temp2 = temp[temp > - corr_threshold]

# betters = []
# for x in pearsons.keys():
#     for y in pearsons.keys():
#         if pearsons[x][y] < corr_threshold and pearsons[x][y] > - corr_threshold:
#             if x not in betters: betters.append(x)
#             if y not in betters: betters.append(y)
# len(betters)

pearsons['Otu0004']['Otu0005'] = 0.9
pearsons['Otu0005']['Otu0004'] = 0.9
# betters = []
# worses = []
keys = pearsons.keys()
flat_keys, flat_values = [], []
for i in range(len(keys)):
    for j in range(i+1, len(keys)):
        x, y = keys[i], keys[j]
        idx = x + '#' + y
        flat_keys.append(idx)
        flat_values.append(pearsons[x][y])
        # if pearsons[x][y] < corr_threshold and pearsons[x][y] > -corr_threshold:
        #     if x not in betters: betters.append(x)
        #     if y not in betters: betters.append(y)
        # else:
        #     pass
        # print(betters)
        
# len(betters)
# s = pd.Series([5.6, 2, 3], index=['a', 'b', 'c'])
# s.sort_values()
# s.append({'d': '9'})
pearsons_flat = pd.Series(flat_values, index=flat_keys).map(lambda x: -x if x < 0 else x)
pearsons_flat.sort_values(ascending=False, inplace=True)
pearsons_filtered = pearsons_flat[pearsons_flat > corr_threshold]

