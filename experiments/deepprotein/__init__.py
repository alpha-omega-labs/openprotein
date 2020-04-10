"""
This file is part of the OpenProtein project.

For license information, please see the LICENSE file in the root directory.
"""

from preprocessing import process_single_raw_data

from models import *
from training import train_model
from util import contruct_dataloader_from_disk, load_model_from_disk,\
    write_out,set_experiment_id,write_prediction_data_to_disk
from experiments.deepprotein.config import *

def run_experiment(parser, use_gpu):
    # parse experiment specific command line arguments
    parser.add_argument('--learning-rate', dest='learning_rate', type=float,
                        default=0.001, help='Learning rate to use during training.')
    parser.add_argument('--embed-size', dest='embed_size', type=int,
                        default=21, help='Embedding size.')
    args, _unknown = parser.parse_known_args()

    all_prediction_data = []
    result_matrices = []
    # pre-process data
    preprocessed_training_file = process_single_raw_data(training_file,use_gpu=use_gpu, force_pre_processing_overwrite=False)
    preprocessed_validation_file = process_single_raw_data(validation_file, use_gpu=use_gpu,
                                                         force_pre_processing_overwrite=False)
    preprocessed_test_file = process_single_raw_data(test_file, use_gpu=use_gpu,
                                                         force_pre_processing_overwrite=False)

    # run experiment

    # model = ExampleModel(args.embed_size, args.minibatch_size, use_gpu=use_gpu)  # embed size = 21
    # model = SimpleRCNN(args.embed_size, args.minibatch_size, use_gpu=use_gpu)  # embed size = 21
    model = DeepResRCNN_100(args.embed_size, args.minibatch_size, use_gpu=use_gpu)  # embed size = 21


    train_loader = contruct_dataloader_from_disk(preprocessed_training_file, args.minibatch_size)
    validation_loader = contruct_dataloader_from_disk(preprocessed_validation_file, args.minibatch_size)

    train_model_path = train_model(data_set_identifier="TRAIN",
                                   model=model,
                                   train_loader=train_loader,
                                   validation_loader=validation_loader,
                                   learning_rate=args.learning_rate,
                                   minibatch_size=args.minibatch_size,
                                   eval_interval=args.eval_interval,
                                   hide_ui=args.hide_ui,
                                   use_gpu=use_gpu,
                                   minimum_updates=args.minimum_updates)

    print(train_model_path)


    # test model
    test_loader = contruct_dataloader_from_disk(preprocessed_test_file, args.minibatch_size)
    write_out("Testing model...")
    model = load_model_from_disk(train_model_path, force_cpu=False)
    _loss, json_data, _ = model.evaluate_model(test_loader)

    all_prediction_data.append(json_data)
    # all_prediction_data.append(model.post_process_prediction_data(prediction_data))
    result_matrix = np.array(json_data['confusion_matrix'])
    result_matrices += result_matrix
    write_out(result_matrix)

    set_experiment_id(
        "TEST-" + str(args.hidden_size) + "-F"
        + str(args.input_data.split(".")[-2]),
        args.learning_rate,
        args.minibatch_size)
    write_out(result_matrices)
    write_prediction_data_to_disk("\n".join(all_prediction_data))