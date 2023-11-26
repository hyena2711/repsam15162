dataset=kvasir
path=Kvasir-SEG


dataset=medsam2d
path=MedSAMDemo_2D
# lr=1000.0
# img_ext=png
# lbl_ext=png
# dataset=task02_heart
# path=Task02_Heart 
# dataset=task04_hippocampus1
# path=Task04_Hippocampus
# dataset=task05_prostate1
# path=Task05_Prostate
# dataset=task09_spleen
# path=Task09_Spleen
# dataset=task10_colon
# path=Task10_Colon
img_ext=png
lbl_ext=png
# img_ext=jpg
# lbl_ext=jpg
lr=1000.0


padding=16
output=output_p${padding}_padding_dice_ce
epochs=1



python test.py --path /path/to/data/root/${path} \
                --checkpoint /path/to/sam_vitb_checkpoint \
                --mrsam ./${output}/${dataset}/ep_${epochs}_lr${lr}/repsam_bbox_p${padding}_final.pth\
                --image_col image_path \
                --mask_col mask_path \
                --use_bbox \
                --img_ext $img_ext \
                --lbl_ext $lbl_ext \
                --pad_size $padding \
                --model_type vit_b \
                --method padding\
                --df ./${output}/${dataset}/ep_${epochs}_lr${lr}/test.csv 
