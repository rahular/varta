declare -A languages=( ["as"]="assamese" ["bh"]="bhojpuri" ["bn"]="bengali" ["gu"]="gujarati" ["hi"]="hindi" ["kn"]="kannada" ["ml"]="malayalam" ["mr"]="marathi" ["ne"]="nepali" ["or"]="oriya" ["pa"]="panjabi" ["ta"]="tamil" ["te"]="telugu" ["ur"]="urdu" )


MODEL_DIR=models/mt5_all_run4
OUTPUT_DIR=models/mt5_xlsum_sum_run4
# evaluate the finetuned model from Varta on xlsum
for lg in bengali english gujarati hindi marathi nepali punjabi sinhala tamil telugu urdu; do
    python run_sum.py   --model_name_or_path=${MODEL_DIR}/checkpoint-41000 \
                        --dataset_name=csebuetnlp/xlsum \
                        --lang=${lg} \
                        --output_dir=${MODEL_DIR} \
                        --max_source_length=512 \
                        --max_target_length=128 \
                        --num_beams=4 \
                        --per_device_eval_batch_size=32 \
                        --ignore_pad_token_for_loss \
                        --do_predict \
                        --predict_with_generate

    mv ${MODEL_DIR}/predictions.txt ${OUTPUT_DIR}/predictions_${lg}.txt
    mv ${MODEL_DIR}/references.txt ${OUTPUT_DIR}/references_${lg}.txt
    mv ${MODEL_DIR}/all_results.json ${OUTPUT_DIR}/all_results_${lg}.json

    echo "Computing ROUGE scores for ${lg}..."

    python -m rouge_score.rouge \
        --target_filepattern=${OUTPUT_DIR}/references_${lg}.txt \
        --prediction_filepattern=${OUTPUT_DIR}/predictions_${lg}.txt  \
        --output_filename=${OUTPUT_DIR}/scores_${lg}.csv \
        --use_stemmer=true \
        --lang="${lg}"
done