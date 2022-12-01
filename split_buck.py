def split_buck(data, buck):
    # get the maximum number in data
    N_max = data[0]

    for i in data:
        if i > N_max:
            N_max = i

    # sort the data w.r.t. data.num_nodes
    sorted_list = sorted(data, key=lambda x: x, reverse=False)
    # print('length of sorted list: ', len(sorted_list))

    # num of elements in each buck
    buck_num = round(len(data) / buck)

    # split point for data w.r.t. buck
    split_pt = []
    cnt = 0
    for i in sorted_list:
        cnt += 1
        if cnt == buck_num:
            cnt = 0
            split_pt.append(i)

    if len(split_pt) == buck:
        split_pt[-1] = N_max + 1
    else:
        split_pt.append(N_max + 1)

    res = []
    for i in data:
        for index, j in enumerate(split_pt):
            if i <= j:
                res.append(index)
                break

    return res
