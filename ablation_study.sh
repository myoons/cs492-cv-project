#!/bin/bash
for thres in {0.9 0.5 0.1 0.05 0.01 0.005 0.001}
do 
    nsml run -d fashion_dataset -g 0 --memo "thres_${thres}" --args "--threshold ${thres}"
done

for (( mu = 0 ; mu < 11 ; mu++ )) ; do
    nsml run -d fashion_dataset -g 0 --memo "mu_${mu}" --args "--mu ${mu}"
done

for name in ["res18", "res50", "dense121"]
do
    nsml run -d fashion_dataset -g 0 --memo "${name}" --args "--name ${name}"
done

for thres in {0.99 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60}
do 
    nsml run -d fashion_dataset -g 0 --memo "thres_${thres}" --args "--threshold ${thres}"
done

for lambda in {4..12}
do
    nsml run -d fashion_dataset -g 0 --memo "labmda_u_${lambda}" --args "--lambda-u ${lambda}"
done

nsml run -d fashion_dataset -g 0 --memo "use_ema_False" --args "--use-ema False"

for optim in ["sgd", "adam"]
do
    nsml run -d fashion_dataset -g 0 --memo "optim_${optim}" --args "--optim ${optim}"
done
