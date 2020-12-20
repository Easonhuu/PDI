# flag=1
# result=1
# while [ "$flag" -eq 1 ]
# do
#     sleep 1s
#     PID=846
#     PID_EXIST=$(ps u | awk '{print $2}'| grep -w $PID)
#     if [ ! $PID_EXIST ]; then
#         echo "process is finished"
#         flag=0
#     fi
# done

# python ./fine-tuning/main.py --gpu_start 0 --n_gpu 2 --multi_dropout_num 1
python ./fine-tuning/main.py --gpu_start 0 --n_gpu 2 --multi_dropout_num 1 --drug_dim 256