import pandas as pd
import os
import shutil
import pdb
data_read=pd.read_csv('./trainLabels.csv')
for row in data_read.itertuples():
    ok = os.path.isfile('/media/dani/0658C3F958C3E591/Kaggle_diabetic/real_dataset/row.image.jpeg')
    print(row)
    # pdb.set_trace()
    if ok:
        if row.level > 0:
            print(row.image)
            shutil.copy2('/media/dani/0658C3F958C3E591/Kaggle_diabetic/real_dataset/row.image.jpeg','/media/dani/0658C3F958C3E591/Kaggle_diabetic/dataset/train/2/row.image.jpeg')
        else:
            print(row.image)
            shutil.copy2('/media/dani/0658C3F958C3E591/Kaggle_diabetic/real_dataset/row.image.jpeg','/media/dani/0658C3F958C3E591/Kaggle_diabetic/dataset/train/1/row.image.jpeg')
    else:
        print("not found")
