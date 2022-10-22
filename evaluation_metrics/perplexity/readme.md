# Perplexity with xlnet

1. Install Huggingface's Transformer repo from:https://github.com/huggingface/transformers

   You can run:

   ```
   git clone https://github.com/huggingface/transformers.git
   cd transformer
   pip install -e .
   pip install datasets
   pip install torch torchvision
   ```

2. Arrange your model generated outputs into txt file line by line

3. Evaluate perplexity with xlnet-base-cased with the following command. The train_file could be some dummy files.

   ```
    python examples/pytorch/language-modeling/run_clm.py \
       --model_name_or_path xlnet-base-cased \
       --train_file YOUR_GENERATED_OUTPUTS \
       --validation_file YOUR_GENERATED_OUTPUTS \
       --do_eval \
       --output_dir YOUR_OUTPUT_FILE \
	   
	   
	python run_clm.py  --model_name_or_path xlnet-base-cased --train_file pew/bertqa_data/generatedAnswers.txt   --validation_file pew/bertqa_data/generatedAnswers.txt  --do_eval    --output_dir pew/bertqa/targetAnswers.txt 
   ```