import torch
import os
from flask import Flask, render_template, request
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer


model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')



def answer_question(question, answer_text):
    '''
    Takes a `question` string and an `answer_text` string (which contains the
    answer), and identifies the words within the `answer_text` that are the
    answer. Prints them out.
    '''
    # ======== Tokenize ========
    # Apply the tokenizer to the input text, treating them as a text-pair.
    input_ids = tokenizer.encode(question, answer_text)

    # Report how long the input sequence is.
    print('Query has {:,} tokens.\n'.format(len(input_ids)))

    # ======== Set Segment IDs ========
    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    # ======== Evaluate ========
    # Run our example question through the model.
    start_scores, end_scores = model(torch.tensor([input_ids]), # The tokens representing our input text.
                                    token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question from answer_text

    # ======== Reconstruct Answer ========
    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Get the string versions of the input tokens.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Start with the first token.
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):
        
        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        
        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]

    return 'Answer: "' + answer + '"'



import textwrap
with open("C:\\Users\\Mustafa\\Desktop\\thesis\\chatbot\\ata.txt","r", encoding = "utf-8") as file:
     bert_abstract = file.read()

wrapper = textwrap.TextWrapper(width=512) 

question1 = "What was Ataturk's jobs?"
question2 = "When was Atatürk born?"
question3 = "Where was Atatürk born??"
question4 ="When Atatürk died?"
question5 = "When did Atatürk become president of Turkey?"
answer1 = answer_question(question1, bert_abstract)
answer2 = answer_question(question2, bert_abstract)
answer3 = answer_question(question3, bert_abstract)
answer4 = answer_question(question4, bert_abstract)
answer5 = answer_question(question5, bert_abstract)
app = Flask(__name__)
STATIC_DIR = os.path.abspath('../static')
@app.route("/")
def yazdir():
    return render_template("index.html",STATIC_DIR = STATIC_DIR, question1= question1, question2= question2,question3= question3,question4= question4,question5= question5 ,answer1=answer1, answer2=answer2, answer3=answer3, answer4 = answer4, answer5=answer5)

if __name__ == "__main__":
    app.run(debug=True)
    
    



