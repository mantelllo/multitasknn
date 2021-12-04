from lib import Engine
import argparse


def train(csv_file, task2_loss_multiplier):
    engine = Engine()
    X, y = engine.load_data(csv_file)
    model = engine.load_model()
    engine.train_model(model, X, y, task2_loss_multiplier)
    engine.print_variables(model)



def get_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--task2-loss-multiplier', type=float, default=0.01,
                        help='Penalization multiplier for loss of task2. Set 0.001 if want '\
                             'to prioritize training for task1 (warning: risk of overfitting).')
    parser.add_argument('--csv-file', type=str, default='data/multitasklearnig_task.csv',
                        help='Input file with columns "DatasetID,x1,x2,x3,x4,x5,x6,z,y1,y2".')

    return parser.parse_args()
    

if __name__ == '__main__':
    args = get_args()
    train(args.csv_file, args.task2_loss_multiplier)
