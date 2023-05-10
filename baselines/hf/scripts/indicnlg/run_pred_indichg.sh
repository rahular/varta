DATSET_NAME=ai4bharat/IndicHeadlineGeneration
MODEL_DIR=models/varta-t5-1M-indichg
TEXT_COLUMN=input
SUMMARY_COLUMN=target
MAX_SOURCE_LEN=512
MAX_TARGET_LEN=64

declare -a LANGS=("as" "bn" "gu" "hi" "kn" "ml" "mr" "or" "pa" "ta" "te")
declare -A languages=( ["as"]="assamese" ["bh"]="bhojpuri" ["bn"]="bengali" ["gu"]="gujarati" ["hi"]="hindi" ["kn"]="kannada" ["ml"]="malayalam" ["mr"]="marathi" ["ne"]="nepali" ["or"]="oriya" ["pa"]="panjabi" ["ta"]="tamil" ["te"]="telugu" ["ur"]="urdu" )

for lg in ${LANGS[@]}; do  # en
    CUDA_VISIBLE_DEVICES=0 python run_sum.py --model_name_or_path=${MODEL_DIR} \
                                            --dataset_name=${DATSET_NAME} \
                                            --text_column=${TEXT_COLUMN} \
                                            --lang=${lg} \
                                            --summary_column=${SUMMARY_COLUMN} \
                                            --output_dir=${MODEL_DIR} \
                                            --max_source_length=${MAX_SOURCE_LEN} \
                                            --max_target_length=${MAX_TARGET_LEN} \
                                            --num_beams=4 \
                                            --per_device_eval_batch_size=32 \
                                            --ignore_pad_token_for_loss \
                                            --do_predict \
                                            --predict_with_generate
    mv ${MODEL_DIR}/predictions.txt ${MODEL_DIR}/predictions_${lg}.txt
    mv ${MODEL_DIR}/references.txt ${MODEL_DIR}/references_${lg}.txt
    mv ${MODEL_DIR}/all_results.json ${MODEL_DIR}/all_results_${lg}.json
    echo "Computing ROUGE scores for ${languages[$lg]}..."
    python -m rouge_score.rouge \
        --target_filepattern=${MODEL_DIR}/references_${lg}.txt \
        --prediction_filepattern=${MODEL_DIR}/predictions_${lg}.txt  \
        --output_filename=${MODEL_DIR}/scores_${lg}.csv \
        --use_stemmer=true \
        --lang="${languages[$lg]}"
done
