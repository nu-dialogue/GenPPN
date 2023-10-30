# Joint BERT NLU for System Agent
The BERT NLU weights originally published in ConvLab-2 ([link](https://github.com/thu-coai/ConvLab-2/tree/master/convlab2/nlu/jointBERT/multiwoz)) are not available in the latest version of transformers, so we have to retrain them.

## Train NLU
1. Prepare dataset and traning config for user utterance understanding model
    ```bash
    python prepare_dataset_and_training_config.py \
        --data_name "full-usr" \
        --mode "usr"
    ```
    This will generate data files of `multiwoz_data/full-usr/*` and training config file of `configs/full-usr.json`.

2. (Optional) Modify the training hyperparameters
    - You can modify the config file `configs/full-usr.json` to change the training hyperparameters.

3. Run training script
    - First, modify the `train_script_path` in `train.sh`
        - `train_script_path`: path to `<absolute_dir>/ppn-nlg/ConvLab-2/convlab2/nlu/jointBERT`
    - Then, run the training script with `configs/full-usr.json`
        ```bash
        chmod +x train.sh
        ./train.sh configs/full-usr.json
        ```
    This will save the trained model to `outputs/full-usr/`.

## (Optional) Test NLU
Test the trained model
- First, modify the `train_script_path` in `test.sh`
    - `train_script_path`: path to `<absolute_dir>/ppn-nlg/ConvLab-2/convlab2/nlu/jointBERT`
- Then, run the test script
    ```bash
    chmod +x test.sh
    ./test.sh configs/full-usr.json
    ```
