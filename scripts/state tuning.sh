
load_model="/home/rwkv/model/rwkv7-g1-1.5b-20250429-ctx4096.pth"
proj_dir='/home/rwkv/JL/out_model/test'
data_file=/home/rwkv/JL/data/roleplay
#/home/rwkv/JL/data/roleplay
n_layer=24
n_embd=2048

micro_bsz=8
epoch_save=1
epoch_steps=200
ctx_len=128

python train.py --load_model $load_model \
--proj_dir $proj_dir --data_file $data_file \
--vocab_size 65536 \
--data_type binidx \
--n_layer $n_layer --n_embd $n_embd \
--ctx_len $ctx_len --micro_bsz $micro_bsz \
--epoch_steps $epoch_steps --epoch_count 10 --epoch_save $epoch_save \
--lr_init 1e-5 --lr_final 1e-5 \
--accelerator gpu --precision bf16 \
--devices 1 --strategy deepspeed_stage_1 --grad_cp 1 \
--my_testing "x070" \
--peft state --op fla


