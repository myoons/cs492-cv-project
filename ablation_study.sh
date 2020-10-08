mus = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
names = ["res18", "res50", "dense121"]

memo = 
nsml run -d fashion_dataset -g 1 -m --num-classes --name --lr --warmup --wdecay --nesterov --nesterov --mu --lambda-u --threshold