#!/bin/bash
for lambda in {4..12}
do
    nsml run -d fashion_dataset -g 0 --memo "labmda_u_${lambda}" --args "--lambda-u ${lambda}"
done

for lr in 0.5 0.05 0.005
do 
    nsml run -d fashion_dataset -g 0 --memo "lr_${lr}" --args "--lr ${lr}"
done

for (( mu = 0 ; mu < 11 ; mu++ )) ; do
    nsml run -d fashion_dataset -g 0 --memo "mu_${mu}" --args "--mu ${mu}"
done

for thres in 0.99 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60
do 
    nsml run -d fashion_dataset -g 1 --memo "thres_${thres}" --args "--threshold ${thres}"
done

for optim in ["sgd", "adam", "adamw"]
do
    nsml run -d fashion_dataset -g 0 --memo "optim_${optim}" --args "--optim ${optim}"
done
