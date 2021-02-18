set -x
export LD_LIBRARY_PATH=/home/vlad/code/dynet/build/dynet/

L=$1
PARSER="./parser-cpu --dynet-seed 42 --dynet-mem 2048 --dynet-autobatch 1 --max-iter 50"
echo "language=$L"
# for L in zh ja ro en
# do
echo "language=$L"
for lr in 1 0.5 0.25 0.125 0.0625
do
    # ${PARSER} --init-lr ${lr} --marginal --no-cost-augment --lang $L > ${L}_mar_per_lr${lr}.txt
    ${PARSER} --init-lr ${lr} --sparsemap --lang $L > ${L}_spa_aug_lr${lr}.txt
    ${PARSER} --init-lr ${lr} --sparsemap --no-cost-augment --lang $L > ${L}_spa_per_lr${lr}.txt
    ${PARSER} --init-lr ${lr} --lang $L > ${L}_map_aug_lr${lr}.txt
    ${PARSER} --init-lr ${lr} --no-cost-augment --lang $L  > ${L}_map_per_lr${lr}.txt
done
# done
