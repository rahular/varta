model=lead1
declare -A languages=( ["en"]="english" ["as"]="assamese" ["bh"]="bhojpuri" ["bn"]="bengali" ["gu"]="gujarati" ["hi"]="hindi" ["kn"]="kannada" ["ml"]="malayalam" ["mr"]="marathi" ["ne"]="nepali" ["or"]="oriya" ["pa"]="panjabi" ["ta"]="tamil" ["te"]="telugu" ["ur"]="urdu" )
for lg in as ar bh bn gu en hi kn ml mr ne or pa ta te ur; do
	python -m rouge_score.rouge \
	        --target_filepattern=${lg}_reference.txt \
	        --prediction_filepattern=${model}_${lg}_prediction.txt  \
	        --output_filename=scores_${lg}.csv \
	        --use_stemmer=true \
	        --lang="${languages[$lg]}"
done
