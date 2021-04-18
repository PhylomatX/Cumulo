import os
import numpy as np
import pickle as pkl
from functools import reduce
from absl import app
from absl import flags

flags.DEFINE_string('path', None, help='Directory where trainings are located')
flags.DEFINE_list('include', None, help='Names (or substrings of names) of trainings which should get added to the table.')
flags.DEFINE_list('exclude', None, help='Names (or substrings of names) of trainings which should get excluded from the table.')
FLAGS = flags.FLAGS


def main(_):
    trainings = os.listdir(FLAGS.path)
    training_names = []
    f1_scores = []
    accuracies = []
    exclude_list = FLAGS.exclude
    if exclude_list is None:
        exclude_list = []
    for training in trainings:
        valid = True
        for exclude in exclude_list:
            if exclude in training:
                valid = False
        if not valid:
            continue
        for include in FLAGS.include:
            if include in training:
                training_scores = []
                training_accuracies = []
                with open(os.path.join(FLAGS.path, f'{training}/predictions/total/total_report.pkl'), 'rb') as f:
                    report = pkl.load(f)
                with open(os.path.join(FLAGS.path, f'{training}/predictions/total/total_matrix.pkl'), 'rb') as f:
                    matrix = pkl.load(f)
                for i in range(matrix.shape[0]):
                    training_scores.append(report[str(i)]['f1-score'])
                    training_accuracies.append(matrix[i, i] / report[str(i)]['support'])
                training_names.append(training)
                f1_scores.append(training_scores)
                accuracies.append(training_accuracies)
                break
    score_table = ''
    accuracy_table = ''
    scores_np = np.array(f1_scores)
    accuracies_np = np.array(accuracies)
    for ix, training in enumerate(training_names):
        training_scores = [f' & \\textbf{{{round(x, 2)}}}' if x == np.max(scores_np[:, col]) else f' & {round(x, 2)}' for col, x in enumerate(f1_scores[ix])]
        training_accuracies = [f' & \\textbf{{{round(x * 100, 2)}}}' if x == np.max(accuracies_np[:, col]) else f' & {round(x * 100, 2)}' for col, x in enumerate(accuracies[ix])]
        score_table += f'\n{training}' + reduce(lambda x, y: x + y, training_scores)
        accuracy_table += f'\n{training}' + reduce(lambda x, y: x + y, training_accuracies)
    result = 'F1-scores:\n' + score_table + '\n\n\nAccuracies:\n' + accuracy_table
    with open(os.path.join(FLAGS.path, 'table.txt'), 'w') as f:
        f.write(result)


if __name__ == '__main__':
    app.run(main)
