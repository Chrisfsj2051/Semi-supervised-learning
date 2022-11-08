CFGS=(

)

source activate ssl

for CFG in ${CFGS[@]} ;
do
  nohup python train.py --c ${CFG} &
done