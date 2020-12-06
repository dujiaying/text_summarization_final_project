
from nltk.translate.bleu_score import sentence_bleu

def sentence_bleu_n(ref, hyp, weights):
  # ref and hyp are strings
  # the weight list corresponds to [bleu-1_weight, bleu-2_weight, bleu-3_weight, bleu-4_weight]
  # nltk calculates up to bleu-4
  return sentence_bleu(references = [ref.split()], 
                       hypothesis = hyp.split(),
                       weights = weights)

# bleu1
pred_output['bleu1'] = pred_output[['y_pred', 'y_true']].apply(lambda x: sentence_bleu_n(x[1], x[0], weights = [1,0,0,0]), axis=1)

# bleu2
pred_output['bleu2'] = pred_output[['y_pred', 'y_true']].apply(lambda x: sentence_bleu_n(x[1], x[0], weights = [0,1,0,0]), axis=1)
