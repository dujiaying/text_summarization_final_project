
from nltk.translate.bleu_score import sentence_bleu

def sentence_bleu_n(ref, hyp, weights):
  # ref and hyp are strings
  # bleu1: weights = [1, 0, 0, 0]
  # bleu2: weights = [.5, .5, 0, 0]
  # bleu3: weights = [.33, .33, .33, 0]
  # bleu3: weights = [.25, .25, .25, .25] <- default in sentence_bleu
  return sentence_bleu(references = [ref.split()], 
                       hypothesis = hyp.split(),
                       weights = weights)

# bleu1
pred_output['bleu1'] = pred_output[['y_pred', 'y_true']].apply(lambda x: sentence_bleu_n(x[1], x[0], weights = [1,0,0,0]), axis=1)

# bleu2
pred_output['bleu2'] = pred_output[['y_pred', 'y_true']].apply(lambda x: sentence_bleu_n(x[1], x[0], weights = [.5,.5,0,0]), axis=1)
