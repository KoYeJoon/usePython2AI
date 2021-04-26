import pandas as pd
import numpy as np
annot = np.loadtxt('/home/kite/yejoon/mot_neural_solver/output/experiments/04-23_11:54_evaluation/mot_files/MOT17-14-FRCNN.txt',delimiter=',',skiprows=0,dtype= float)
annot[:,1] = np.ones(len(annot),dtype=int)

df = pd.DataFrame(annot)
df[0] = df[0].astype(int)
df[1] = df[1].astype(int)
df[6]= df[6].astype(int)
df[7]= df[7].astype(int)
df[8]= df[8].astype(int)
df[9]= df[9].astype(int)
print(df)

# index 는 말그대로 index, header는 column 제외하고 저장하고 싶은 경우
df.to_csv('label_remove.txt',sep=',', index =False, header=False)