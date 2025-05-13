import torch
import argparse
import time
import os
import sys
import glob

import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from numpy.random import RandomState

from TopoEEG import \
    modeling_TopoEEG, \
    modeling_classifier_base, \
    train, \
    PSD_calculator, \
    topo1_persistent_homology, \
    topo2_nework_topology, \
    reg1_power_spectrum_analysis_AR, \
    reg2_connectivity_analysis_CORR

frequency_domains = {
                    'delta' : (0.9, 4.),
                    'theta' : (4. , 8.),
                    'alpha' : (8. , 14.),
                    'beta'  : (14., 25.),
                    'gamma' : (25., 40.),
                    'all'   : (0.5, 40.)
                    }
topo_labels = {
    'topo1': 'Topo1. Persistent Homology',
    'topo2': 'Topo2. Network Topology ',
    'reg1': 'Reg1. Power spectrum analysis, AR',
    'reg2': 'Reg2. Connectivity analysis, CORR'
               }

def main(
    freq_data_path,
    n_psd_output,
    domain,
    data_ratio,
    val_size,
    test_size,
    feature_types,
    perform_train,
    save_features_for_later,
    random_state,
    num_epochs,
    batch_size
    args
):
    rng = RandomState(random_state) 

    print(args)
    print(f'Complex feature extraction in domain: {domain}')

    # Get freq data
    input_data = pd.read_pickle(freq_data_path) 

    # Change split
    # # Even classes
    print('Pre even classes:\t', input_data.shape)
    input_data = input_data[np.isin(input_data.asmr_type, args.positive_class) | np.isin(input_data.asmr_type, args.negative_class)].reset_index(drop=True)
    print('Post even classes:\t', input_data.shape)


    # # Get smaller sample (if needed, data_ratio < 1)
    if data_ratio < 1:
        ids = rng.choice(np.arange(input_data.shape[0]), int(input_data.shape[0]*data_ratio), replace=False)
    else:
        ids = np.arange(len(input_data))
    print('Post data sampling:\t', len(ids))

    # # Split
    X = np.array([list(np.vstack(x)) for x in tqdm(input_data.loc[ids, args.channels].values, desc='compile X')])
    y = np.zeros(len(input_data))
    y[np.isin(input_data.asmr_type, args.positive_class)] = 1 # 2!!! class classification
    y = y[ids]

    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(X, y, ids, test_size=val_size, stratify=y, random_state=random_state)
    X_val, X_test, y_val, y_test, ids_val, ids_test = train_test_split(X_val, y_val, ids_val, test_size=test_size, stratify=y_val, random_state=random_state)

    freq_sample = X_train[0,0,:]
    psd_calculator = PSD_calculator.PSD_calculator(freq_sample, frequency_domains, n_psd_output)

    topo_transfomers = {'topo1': topo1_persistent_homology.PersistentHomologyTransformer(psd_calculator=psd_calculator,
                                                                                         domain=domain,
                                                                                         one_time_series_len=n_psd_output,
                                                                                         n_long_living=args.topo1_n_long_living),
                        'topo2': topo2_nework_topology.NetworkTopologyTransformer(),
                        'reg1': reg1_power_spectrum_analysis_AR.ARTransformer(psd_calculator=psd_calculator, domain=domain, order=args.reg1_order),
                        'reg2': reg2_connectivity_analysis_CORR.CorrelationTransformer(psd_calculator=psd_calculator, domain=domain)
                        }

    topo_X_train = {}
    topo_X_val = {}
    topo_X_test = {}

    input_shapes = {}

    feature_extraction_start = time.perf_counter()

    for feature_type in set(feature_types) & set(topo_transfomers.keys()) :
        print(f'Current features:\t{topo_labels[feature_type]}')
        file_save_id =  '_'.join( [domain, 
                                   str(random_state),
                                   '_'.join(ids_val[-5:].astype(str)), 
                                   '_'.join(ids_test[-5:].astype(str)), 
                                   '_'.join(ids_train[-5:].astype(str))] )
        file_name = feature_type + file_save_id + '.pt'
        
        pattern = '*'+file_name
        if feature_type == 'reg1':
            file_name = str(args.reg1_order) + file_name
        elif feature_type == 'topo1':
            file_name = str(args.topo1_n_long_living) + file_name
        file_dir = os.path.join(os.path.dirname(freq_data_path), file_save_id)

        if \
        os.path.exists(os.path.join(file_dir, 'train_'+file_name)) and \
        os.path.exists(os.path.join(file_dir, 'val_'+file_name)) and \
        os.path.exists(os.path.join(file_dir, 'test_'+file_name)):
            print('\tLoading features from', file_dir)
            topo_X_train[feature_type] = torch.load(os.path.join(file_dir, 'train_'+ file_name), weights_only=True)
            topo_X_val[feature_type] = torch.load(os.path.join(file_dir, 'val_'+ file_name), weights_only=True)
            topo_X_test[feature_type] = torch.load(os.path.join(file_dir, 'test_'+ file_name), weights_only=True)
        else:
            curr_feature_extraction_start = time.perf_counter()

            topo_transfomer = topo_transfomers[feature_type]

            topo_transfomer.fit(X_train)
            topo_X_train[feature_type] = torch.from_numpy(topo_transfomer.transform(X_train)).float()
            topo_X_val[feature_type] = torch.from_numpy(topo_transfomer.transform(X_val)).float()
            topo_X_test[feature_type] = torch.from_numpy(topo_transfomer.transform(X_test)).float()

            curr_feature_extraction_end = time.perf_counter()
            print(f"\tFeatures extracted in {curr_feature_extraction_end - curr_feature_extraction_start:.6f} seconds")
            if save_features_for_later:
                print('\tSaving features to', file_dir)
                os.makedirs(file_dir, exist_ok=True) 
                # for filepath in glob.glob(os.path.join(file_dir, pattern)):
                #     os.remove(filepath)
                torch.save(topo_X_train[feature_type], os.path.join(file_dir, 'train_'+file_name))
                torch.save(topo_X_test[feature_type], os.path.join(file_dir, 'test_'+file_name))
                torch.save(topo_X_val[feature_type], os.path.join(file_dir, 'val_'+file_name))

        input_shapes[feature_type] = topo_X_test[feature_type][0].shape
        print(f'\tFeatures per entry:', input_shapes[feature_type])


    feature_extraction_end = time.perf_counter()
    print(f"All features extracted in {feature_extraction_end - feature_extraction_start:.6f} seconds")

    # Model
    topoEEG_model = modeling_TopoEEG.TopoEEG(args=args,
                                             input_shapes=input_shapes,
                                             classifier_base=modeling_classifier_base.Classifier
                                             )

    # Train
    if perform_train:
        train_start = time.perf_counter()
        topoEEG_model = train.train(topoEEG_model,
                                    topo_X_train, y_train,
                                    topo_X_val, y_val,
                                    num_epochs,
                                    batch_size)
        train_end = time.perf_counter()
        print(f"Model trained in {train_end - train_start:.6f} seconds")

    # Eval
    eval_start = time.perf_counter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    topoEEG_model.to(device)
    topoEEG_model.eval()

    with torch.no_grad():
        outputs = topoEEG_model(topo_X_test)
        probabilities = torch.sigmoid(outputs) 
        predictions = (probabilities > 0.5).float() 
    predictions = predictions.cpu().numpy()

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    eval_end = time.perf_counter()

    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test F1 Score: {f1:.4f}')
    print(f"Model evaluated in {eval_end - eval_start:.6f} seconds")
    print()


if __name__ == "__main__":
    hostname = os.uname().nodename

    if hostname == "sms":
        print("This script cannot be run on the login server. Use sbatch!")
        sys.exit(1)


    parser = argparse.ArgumentParser(
        description='Process PSD data for specific frequency domains.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Run args
    parser.add_argument(
        '--freq_data_path', 
        type=str, 
        default='/home/evlebedyuk/asmr/data/compressed/event_data_NOT_AVERAGED.pkl',
        help='Path to the frequencies data pickle file'
    )
    parser.add_argument(
        '--domain', 
        type=str, 
        default='all',
        choices=list(frequency_domains.keys()),
        help='Target frequency domain'
    )
    parser.add_argument(
        '--n_psd_output', 
        type=int, 
        default=800,
        help='Desired PSD arrays length'
    )
    parser.add_argument(
        '--data_ratio', 
        type=float, 
        default=1.,
        help='Used data ratio'
    )
    parser.add_argument(
        '--val_size', 
        type=float, 
        default=0.3,
        help='Validation size'
    )
    parser.add_argument(
        '--test_size', 
        type=float, 
        default=0.5,
        help='Test size'
    )
    parser.add_argument(
        '--feature_types', 
        type=str, 
        nargs='+', 
        default=[
                'topo1',
                'topo2',
                'reg1',
                'reg2'
                ],
        help='Feature types'
    )
    parser.add_argument(
        '--perform_train', 
        type=bool, 
        default=True,
        help='Does model need training?'
    )
    parser.add_argument(
        '--num_epochs', 
        type=int, 
        default=100,
        help='Number of train epochs'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=32,
        help='Size of train batch'
    )
    parser.add_argument(
        '--save_features_for_later', 
        type=bool, 
        default=False,
        help='Save resulting features?'
    )

    # Model args
    parser.add_argument(
        '--channels', 
        type=str, 
        nargs='+', 
        default=[
                'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3',
                'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1',
                'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz',
                'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4',
                'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4',
                'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8',
                'P10', 'PO8', 'PO4', 'O2'
                ],
        help='Space-separated list of target EEG channels'
    )
    parser.add_argument(
        '--all_classes', 
        type=str, 
        nargs='+', 
        default=[
                'B1(Pre_Baseline)',
                'B2(Pre_Relaxed)',
                'B3(WeakASMR)',
                'B4(StrongASMR)',
                'B5(Post_Relaxed)',
                'B6(Post_Baseline)',
                ],
        help='Space-separated list of all classes'
    )
    parser.add_argument(
        '--positive_class', 
        type=str, 
        nargs='+', 
        default=['B4(StrongASMR)'],
        help='Space-separated list of postitive class lables'
    )
    parser.add_argument(
        '--negative_class', 
        type=str, 
        nargs='+', 
        default=['B1(Pre_Baseline)', 'B2(Pre_Relaxed)'],
        help='Space-separated list of postitive class lables'
    )
    parser.add_argument(
        '--topo1_n_long_living', 
        type=int, 
        default=30,
        help='N long living'
    )
    parser.add_argument(
        '--reg1_order', 
        type=int, 
        default=20,
        help='AR order'
    )
    parser.add_argument(
        '--fusion_method', 
        type=str, 
        default='feature_level',
        choices=['feature_level', 'score_level', 'decision_level'],
        help='Fusion method'
    )
    parser.add_argument(
        '--classifier_base_n_out', 
        type=int, 
        default=8,
        help='Classifier base n_out'
    )
    parser.add_argument(
        '--random_state', 
        type=int, 
        default=42,
        help='Random state'
    )
    parser.add_argument(
        '--topo1_weight', 
        type=float, 
        default=1/4,
        help='Topo1 weight'
    )
    parser.add_argument(
        '--topo2_weight', 
        type=float, 
        default=1/4,
        help='Topo2 weight'
    )
    parser.add_argument(
        '--reg1_weight', 
        type=float, 
        default=1/4,
        help='Reg1 weight'
    )
    parser.add_argument(
        '--reg2_weight', 
        type=float, 
        default=1/4,
        help='Reg2 weight'
    )
    
    args = parser.parse_args()

    result = main(
        freq_data_path=args.freq_data_path,
        n_psd_output=args.n_psd_output,
        domain=args.domain,
        data_ratio=args.data_ratio,
        val_size=args.val_size,
        test_size=args.test_size,
        feature_types=args.feature_types,
        perform_train=args.perform_train,
        save_features_for_later=args.save_features_for_later,
        random_state=args.random_state,
        args = modeling_TopoEEG.ModelArgs(args)
    )
