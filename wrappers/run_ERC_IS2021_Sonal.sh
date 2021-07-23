


for i in 1 2 3 4 5; do
    bash run_v2_ERC.sh $i E 1 24 5 0 42 True UttClassif_ResNet
done

exit


for i in 1 2 3 4 5 #3 #5 # 3#1
do
    bash run_v2_ERC.sh $i E 1 24 5 0 42 True UttClassif_bilstm 
done

exit
sonal_dir=$1  # /export/c12/sjoshi/RAGHU/exp/

# check for Sonal's jobs \qstat -u sjoshi

source /home/rpapagari/.bashrc
cd /export/c02/rpapagari/daseg_erc
source activate daseg_v2

for i in 2 3 4 5 #1
do
    bash run_v2_ERC_temp.sh $i TE 1 24 5 0 42 True UttClassif_bilstm $sonal_dir
    #bash run_v2_ERC_temp.sh $i TE 0 24 5 0 42 False UttClassif_bilstm $sonal_dir
done

