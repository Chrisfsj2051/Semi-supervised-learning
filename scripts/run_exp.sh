CFGS=(
'config/classic_cv/fixmatch/fixmatch_cifar10_40_1gpu.yaml'
'config/classic_cv/fixmatch/fixmatch_cifar10_40_8gpu.yaml'
)

source activate ssl

for CFG in ${CFGS[@]} ;
do
  nohup python train.py --c ${CFG} &
done