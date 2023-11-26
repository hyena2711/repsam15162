dataset=medsam2d
path=MedSAMDemo_2D
# lr=1000.0
# img_ext=png
# lbl_ext=png
# dataset=task02_heart
# path=Task02_Heart
# dataset=task04_hippocampus
# path=Task04_Hippocampus
# dataset=task05_prostate_lbl1
# path=Task05_Prostate
# dataset=task09_spleen
# path=Task09_Spleen
# dataset=task10_colon
# path=Task10_Colon
img_ext=png
lbl_ext=png
# path=ISIC
# dataset=isic
# img_ext=jpg
# lbl_ext=png
lr=1000.0



dataset=$1
path=$2
img_ext=$3
lbl_ext=$4
padding=$5
label_id=$6


if [ "$label_id" -eq 0 ]; then
    output=output_p${padding}
    python repsam.py --path /path/data/root/directory/${path} \
                --checkpoint /path/to/sam_vitb_checkpoint \
                --mode train \
                --image_col image_path \
                --mask_col mask_path \
                --img_ext $img_ext \
                --lbl_ext $lbl_ext \
                --model_type vit_b \
                --num_epochs 1 \
                --use_bbox \
                --lr $lr \
                --batch_size 4 \
                --method padding \
                --loss_name dice_ce \
                --num_gpus 0 1 \
                --pad_size $padding \
                --dataset_name $dataset\
                --output_dir $output
else
    output=output_precise_bbox_p${padding}

    python repsam.py --path /path/data/root/directory/${path} \
                --checkpoint /path/to/sam_vitb_checkpoint \
                --mode train \
                --image_col image_path \
                --mask_col mask_path \
                --img_ext $img_ext \
                --lbl_ext $lbl_ext \
                --model_type vit_b \
                --num_epochs 1 \
                --use_bbox \
                --lr $lr \
                --batch_size 4 \
                --method padding \
                --loss_name dice_ce \
                --num_gpus 0 1 \
                --pad_size $padding \
                --dataset_name $dataset\
                --output_dir $output \
                --label_id $label_id
fi