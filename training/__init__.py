"""Prediction stuff here"""

from .adjusted_close import run_training_loop
from .close import run_training_loop as run_training_loop_close
import threading

hparams = {
        'HP_LSTM_UNITS': 400,
        'HP_DROPOUT': 0.3,
        'HP_EPOCHS': 200
    }

if __name__ == "__main__":
    close_thread = threading.Thread(target=run_training_loop_close, args=(hparams,))
    close_thread.setName('Close Model Training')
    close_thread.start()

    adj_close_thread = threading.Thread(target=run_training_loop, args=(hparams,))
    adj_close_thread.setName('Adjusted Close Model Training')
    adj_close_thread.start()


    close_thread.join()
    adj_close_thread.join()