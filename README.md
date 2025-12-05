# ID2223 Lab2 Group 2333

### Link to Hugging Face Spaces: https://huggingface.co/spaces/Yuxin020807/Iris

## Task 1
Fine-Tune a pre-trained large language (transformer) model and build a serverless UI for using that model

a. Fine-tune an existing pre-trained large language model on the FineTome Instruction Dataset

b. Build and run an inference pipeline with a Gradio UI on Hugging Face Spaces for your model.

## Task 2
Describe in your README.md program ways in which you can improve model performance using

(a) model-centric approach - e.g., tune hyperparameters, change the fine-tuning model architecture, etc

(b) data-centric approach - identify new data sources that enable you to train a better model that one provided in the blog post

1. If you can show results of improvement, then you get the top grade.
2. Try out fine-tuning a couple of different open-source foundation LLMs to get one that works best with your UI for inference (inference will be on CPUs, so big models will be slow).
3. You are free to use other fine-tuning frameworks, such as Axolotl of HF FineTuning - you do not have to use the provided unsloth notebook.

# Ways in which one can improve model performance
### Model-Centric Approach
- **Hyper Parameters Tunning:** Tuning hyperparameters such as learning rate, weight decay, and number of warmup steps can significantly improve model performance and stability. We decided to increase the number of warmup steps with considerations of the total number of update steps. 
- **Choosing the Right Model:** Many deployment platforms struggle with bnb-4bit models, so we avoided those where compatibility was an issue. However, since this project specifically targets efficient 4-bit training, we selected a model that supports 4-bit quantization while still providing strong base performance.
- **Using Pre-trained Models:** When training time or data is limited, Pre-trained models are not always the best option. For short or lightweight fine-tuning runs, the marginal improvement from heavily instruction-aligned models may be small compared to the added complexity. Therefore, it can be more effective to start from a general pre-trained base model and fine-tune it on a well-targeted dataset.
### Data-Centric Approach
- **Selecting Dataset:** We used the FineTome-100k dataset because it focuses on educational content, which matches our intended application. For other goals, a different domain-specific dataset would likely yield better performance.
- **Introducing Training and Validation Set:** Splitting the data into training and validation sets allows us to properly measure and improve performance. The training set is used to update the model while the validation set is used for evaluation. Using datasets that come with predefined splits can save time and help ensure a fair comparison of different models and settings.
