from data.toy_dataset import data, preferences
from models.reward_model import RewardModel
from utils.tokenizer import build_vocab, tokenize_text
from training.preference_loss import preference_loss


all_texts_for_vocab = data + preferences

vocab = build_vocab(all_texts_for_vocab)

vocab_size = len(vocab)

print("Vocabulary Size:",vocab_size)


reward_model = RewardModel(vocab_size)

total_loss = 0
num_preferences = 0


for pref in preferences:

    chosen_text = pref['chosen']
    rejected_text = pref['rejected']

    chosen_ids = tokenize_text(chosen_text,vocab)
    rejected_ids = tokenize_text(rejected_text,vocab)

    r_chosen = reward_model(chosen_ids)
    r_rejected = reward_model(rejected_ids)

    loss = preference_loss(r_chosen,r_rejected)

    total_loss += loss.item()
    num_preferences += 1

    print("Prompt:",pref['prompt'])

    print("Chosen:",chosen_text,"score:",r_chosen.item())

    print("Rejected:",rejected_text,"score:",r_rejected.item())

    print("Loss:",loss.item(),"\n")


average_loss = total_loss/num_preferences

print("Average loss:",average_loss)
