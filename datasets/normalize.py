import numpy as np 
from sklearn.preprocessing import StandardScaler 
from timeit import default_timer as dt 

begin = dt()
print('1')
train = np.load('/A/VSE/data/coco_tensor_precomp/train_ims.npy')
end = begin - dt()
print('Done')
exit()
train = train.mean(-1).mean(-1)
scaler = StandardScaler()
train = scaler.fit_transform(train)
print(train.shape)
print('2')
valid = np.load('/A/VSE/data/coco_tensor_precomp/val_ims.npy')
valid = valid.mean(-1).mean(-1)
valid = scaler.transform(valid)
print(valid.shape)
print('3')
test = np.load('/A/VSE/data/coco_tensor_precomp/test_ims.npy')
test = test.mean(-1).mean(-1)
test = scaler.transform(test)
print(test.shape)
print('4')

ftest = np.load('/A/VSE/data/f30k_tensor_precomp/test_ims.npy')
ftest = ftest.mean(-1).mean(-1)
ftest = scaler.transform(ftest)
print(ftest.shape)
print('5')

fval = np.load('/A/VSE/data/f30k_tensor_precomp/val_ims.npy')
fval = fval.mean(-1).mean(-1)
fval = scaler.transform(fval)
print(fval.shape)
print('6')

ftrain = np.load('/A/VSE/data/f30k_tensor_precomp/train_ims.npy')
ftrain = ftrain.mean(-1).mean(-1)
ftrain = scaler.transform(ftrain)
print(ftrain.shape)
print('7')

np.save('/A/VSE/data/coco_precomp/train_ims.npy', train)
np.save('/A/VSE/data/coco_precomp/val_ims.npy', valid)
np.save('/A/VSE/data/coco_precomp/test_ims.npy', test)

print('8')

np.save('/A/VSE/data/f30k_precomp/train_ims.npy', ftrain)
np.save('/A/VSE/data/f30k_precomp/val_ims.npy', fval)
np.save('/A/VSE/data/f30k_precomp/test_ims.npy', ftest)

