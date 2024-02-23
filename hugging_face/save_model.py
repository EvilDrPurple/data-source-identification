from transformers import TFDistilBertForSequenceClassification
from transformers import AutoTokenizer
from huggingface_hub import create_branch
#from huggingface_hub import Repository

#model = TFDistilBertForSequenceClassification.from_pretrained("./test_trainer_1.1", num_labels=28, from_pt=True)
#model.save_pretrained("/home/kylie/Programming/data-source-identification/ml_testing/test_trainer_1.1/model_tf")
#model.push_to_hub("PDAP/url-classifier-test")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenizer.push_to_hub("PDAP/url-classifier-test")

#repo = Repository("PDAP/url-classifier-test", revision="1.1")
#repo.push_to_hub(model)

#create_branch("PDAP/url-classifier-test", branch="1.1")
