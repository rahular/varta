TEST_DIR=test

MODEL_DIR=models/mt5_hi_run3

echo "Doing prediction for ${MODEL_DIR}..."
declare -A languages=( ["as"]="assamese" ["bh"]="bhojpuri" ["bn"]="bengali" ["gu"]="gujarati" ["hi"]="hindi" ["kn"]="kannada" ["ml"]="malayalam" ["mr"]="marathi" ["ne"]="nepali" ["or"]="oriya" ["pa"]="panjabi" ["ta"]="tamil" ["te"]="telugu" ["ur"]="urdu" )

for lg in as bh bn en gu hi kn ml mr ne or pa ta te ur; do
	python run_sum.py   --model_name_or_path=${MODEL_DIR} \
                        --text_column=text \
                        --summary_column=headline \
                        --test_file=${TEST_DIR}/test_${lg}.json \
                        --output_dir=${MODEL_DIR} \
                        --max_source_length=512 \
                        --max_target_length=64 \
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
