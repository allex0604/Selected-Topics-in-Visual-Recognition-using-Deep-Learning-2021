import h5py
import pandas as pd


def get_all_data(hdf5_data, length):
    data_file = pd.DataFrame(
        [],
        columns=['img_name', 'height', 'left', 'top', 'width', 'label']
    )

    for index in range(length):
        # name is a object
        name = hdf5_data['digitStruct']['name'][index].item()
        # one index have 5 integers
        str = hdf5_data[name].value
        # use ASCII code change 5 integers to correspond symbol
        str1 = ''.join([chr(v.item()) for v in str])
        # print(str1)
        item = hdf5_data['digitStruct']['bbox'][index].item()
        lst = {'filename': str1}
        for x in ['height', 'left', 'top', 'width', 'label']:
            attr = hdf5_data[item][x]
            value = [hdf5_data[attr.value[i].item()].value[0][0]
                     for i in range(len(attr))] if len(
                     attr) > 1 else [attr.value[0][0]]
            lst[x] = value
        # key load from columns(important read name next height ,....)
        # if load from index(read name , height(all) , ...)
        data_file = pd.concat([data_file,
                               pd.DataFrame.from_dict(lst, orient='columns')])
        # print(lst)
    data_file.to_csv("train1.csv")

if __name__ == "__main__":
    labels_file = 'train/digitStruct.mat'
    f = h5py.File(labels_file, 'r')
    length = len(f['digitStruct']['name'])
    get_all_data(f, length)
