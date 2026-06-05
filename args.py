import argparse

def get_args():
    parser = argparse.ArgumentParser('Train.')

    # General args
    parser.add_argument('--dataset',
                        type=str,
                        default="TUSZ",
                        choices=("TUSZ", "CHBMIT"),
                        help='Dataset to use.')
    parser.add_argument('--device',
                        type=str,
                        default="cuda",
                        help='Device to run the model.')
    parser.add_argument('--save_dir',
                        type=str,
                        default="results",
                        help='Directory to save the outputs and checkpoints.')
    parser.add_argument(
        '--load_model_path',
        type=str,
        default=None,
        help='Model checkpoint to start training/testing from.')
    parser.add_argument('--test',
                        default=False,
                        action='store_true',
                        help='Whether perform training.')
    parser.add_argument('--rand_seed',
                        type=int,
                        default=123,
                        help='Random seed.')
    parser.add_argument(
        '--task',
        type=str,
        default='detection',
        choices=(
            'detection',
            'classification',
            'prediction',
            'SS pre-training'),
        help="Seizure detection, Seizure prediction, seizure type classification, \
                            or SS pre-training.")
    parser.add_argument('--fine_tune',
                        default=False,
                        action='store_true',
                        help='Whether to fine-tune pre-trained model.')
    # Information Bottleneck
    parser.add_argument('--IB_size', type=int, default=64,
                    help='Size of the information bottleneck latent vector')
    # Input args
    parser.add_argument(
        '--graph_type',
        choices=(
            'individual',
            'combined',
            'dynamic'),
        default='dynamic',
        help='Whether use individual graphs (cross-correlation) or combined graph (distance).')
    parser.add_argument('--max_seq_len',
                        type=int,
                        default='12',
                        help='Maximum sequence length in seconds.')
    parser.add_argument(
        '--output_seq_len',
        type=int,
        default=12,
        help='Output seq length for SS pre-training, in seconds.')
    parser.add_argument('--time_step_size',
                        type=int,
                        default=1,
                        help='Time step size in seconds.')
    parser.add_argument('--input_dir',
                        type=str,
                        default="processed_data",
                        help='Dir to resampled EEG signals (.h5 files).')
    parser.add_argument('--raw_data_dir',
                        type=str,
                        #default="TUSZ_v1.5.2",
                        default='TUSZ_small',
                        help='Dir to TUH data with raw EEG signals.')
    parser.add_argument('--preproc_dir',
                        type=str,
                        default=None,
                        help='Dir to preprocessed (Fourier transformed) data.')
    parser.add_argument('--chb_marker_dir',
                        type=str,
                        default='data/file_markers_chb_mit',
                        help='Directory containing CHB-MIT train/dev/test marker files.')
    parser.add_argument('--save_adj_debug',
                        default=False,
                        action='store_true',
                        help='Save per-sample adjacency matrices for debugging.')
    parser.add_argument(
        '--top_k',
        type=int,
        default=3, #default=3 for original version
        help='Top-k neighbors of each node to keep, for graph sparsity.')

    # Model args
    parser.add_argument("--model_name", type=str, default="evobrain", choices=("evobrain", "lstm", "cnnlstm", "dcrnn", "evolvegcn", "BIOT", "neurogrip"))
    parser.add_argument('--num_nodes',
                        type=int,
                        default=19,
                        help='Number of nodes in graph.')
    parser.add_argument('--num_rnn_layers',
                        type=int,
                        default=2,
                        help='Number of RNN layers in encoder and/or decoder.')
    parser.add_argument(
        '--pretrained_num_rnn_layers',
        type=int,
        default=3,
        help='Number of RNN layers in encoder and decoder for SS pre-training.')
    parser.add_argument('--rnn_units',
                        type=int,
                        default=64,
                        help='Number of hidden units.')
    parser.add_argument('--dcgru_activation',
                        type=str,
                        choices=('relu', 'tanh'),
                        default='tanh',
                        help='Nonlinear activation used in DCGRU cells.')
    # parser.add_argument('--input_dim',
    #                     type=int,
    #                     default=100,
    #                     help='Input seq feature dim.')
    parser.add_argument(
        '--num_classes',
        type=int,
        default=1,
        help='Number of classes for seizure detection/classification.')
    parser.add_argument('--output_dim',
                        type=int,
                        default=100,
                        help='Output seq feature dim.')
    parser.add_argument('--max_diffusion_step',
                        type=int,
                        default=2,
                        help='Maximum diffusion step.')
    parser.add_argument('--cl_decay_steps',
                        type=int,
                        default=3000,
                        help='Scheduled sampling decay steps.')
    parser.add_argument(
        '--use_curriculum_learning',
        default=False,
        action='store_true',
        help='Whether to use curriculum training for seq-seq model.')
    parser.add_argument(
        '--use_fft',
        default=True,
        action='store_true',
        help='Whether the input data is Fourier transformed EEG signal or raw EEG.'
        )

    parser.add_argument('--lambda1',
        type=float,
        default=0.5,
        help='IB loss: I(A;Z)'
        )
    parser.add_argument('--lambda2',
        type=float,
        default=0.5,
        help='IB loss: I(A;Y)'
        )
    
    parser.add_argument('--lambda3',
        type=float, 
        default=0.5,
        help='Temporal Smoothness'
        )

    parser.add_argument('--lambda4',
        type=float,
        default=0.5,
        help='Masked Feature Reconstruction'
        )
    
    parser.add_argument(
        '--embed_dim',
        type=int,
        default=64,
        help='Embed dimension'
        )

    parser.add_argument('--input_dim',
        type=int,
        default=100,
        help='Input EEG signal dimension per node'
        )

    # Training/test args
    parser.add_argument('--train_batch_size',
                        type=int,
                        default=16,#default=128,
                        help='Training batch size.')
    parser.add_argument('--test_batch_size',
                        type=int,
                        default=32, #default=256,
                        help='Dev/test batch size.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=8,
                        help='Number of sub-processes to use per data loader.')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.0,
                        help='Dropout rate for dropout layer before final FC.')
    parser.add_argument('--eval_every',
                        type=int,
                        default=1,
                        help='Evaluate on dev set every x epoch.')
    parser.add_argument(
        '--metric_name',
        type=str,
        default='auroc',
        choices=(
            'F1',
            'acc',
            'loss',
            'auroc'),
        help='Name of dev metric to determine best checkpoint.')
    parser.add_argument('--lr_init',
                        type=float,
                        default=3e-4,
                        help='Initial learning rate.')
    parser.add_argument('--l2_wd',
                        type=float,
                        default=5e-4,
                        help='L2 weight decay.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=100,
                        help='Number of epochs for training.')
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=5.0,
                        help='Maximum gradient norm for gradient clipping.')
    parser.add_argument('--metric_avg',
                        type=str,
                        default='weighted',
                        help='weighted, micro or macro.')
    parser.add_argument('--data_augment',
                        default=True,
                        action='store_true',
                        help='Whether perform data augmentation.')
    parser.add_argument(
        '--patience',
        type=int,
        default=5,
        help='Number of epochs of patience before early stopping.')
    parser.add_argument(
        '--agg',
        type=str,
        default="max",
        help='Aggregation method for node features for graph classification.')
    
    ##--------------------------START---------------------------------
    # NeuroGRIP Configuration
    parser.add_argument('--use_neurogrip', dest='use_neurogrip', action='store_true',
                        help='Enable NeuroGRIP for knowledge-based graph refinement')

    # === Knowledge Base & Retrieval ===
    parser.add_argument('--kg_triplets_path', type=str, default='model/neurogrip/KG_triplets_flat.json',
                        help='Path to flat, FAISS-aligned triplets generated by build_biobert_faiss.py')
    parser.add_argument('--faiss_index_path', type=str, default='model/neurogrip/faiss.index',
                        help='Path to the FAISS index built from knowledge embeddings')
    parser.add_argument('--rag_backend', type=str, default='auto', choices=('auto', 'faiss', 'numpy'),
                        help='Retrieval backend. auto uses FAISS if available, otherwise deterministic NumPy fallback.')
    parser.add_argument('--topk_retrieval', type=int, default=5,
                        help='Top-K knowledge entries to retrieve per EEG node query')

    # === Semantic Query Projection ===
    parser.add_argument('--semantic_input_dim', type=int, default=64,
                        help='Input feature dimension for EEG semantic query projection')
    parser.add_argument('--semantic_hidden_dim', type=int, default=128,
                        help='Hidden layer dimension for semantic projection network')
    parser.add_argument('--semantic_proj_dim', type=int, default=1536,
                        help='Projected semantic dimension for query vectors; BioBERT [head||tail] is 1536 by default')
    parser.add_argument('--semantic_k_hop', type=int, default=2,
                        help='k-hop neighborhood size used by SemAlignQuery')

    # === Graph Refinement Parameters ===
    parser.add_argument('--refine_threshold', type=float, default=0.6,
                        help='Edge confidence threshold for pruning during graph refinement')
    parser.add_argument('--refine_alpha', type=float, default=0.7,
                        help='Fusion coefficient between learned adjacency and knowledge-based adjacency')
    parser.add_argument('--match_mode', type=str, default='evidence', choices=('strict', 'soft', 'evidence'),
                        help='Triplet-edge match policy for knowledge confidence scoring.')
    parser.add_argument('--refine_save_path', type=str, default='./results/graph_refinement',
                        help='Directory to save triplet support logs and refined graphs')
    parser.add_argument('--disable_refine_logging', action='store_true',
                        help='Disable triplet support JSON logging during refinement.')

    # === Integration / Execution ===
    parser.add_argument('--refine_interval', type=int, default=1,
                        help='Run NeuroGRIP refinement every N time steps during training')

    ##--------------------------END------------------------------
    args = parser.parse_args()

    # which metric to maximize
    if args.metric_name == 'loss':
        # Best checkpoint is the one that minimizes loss
        args.maximize_metric = False
    elif args.metric_name in ('F1', 'acc', 'auroc'):
        # Best checkpoint is the one that maximizes F1 or acc
        args.maximize_metric = True
    else:
        raise ValueError(
            'Unrecognized metric name: "{}"'.format(
                args.metric_name))

    # must provide load_model_path if testing only
    if (args.load_model_path is None) and (args.test):
        raise ValueError(
            'For evaluation only, please provide trained model checkpoint in argument load_model_path.')

    # filter type for dcrnn
    if args.graph_type == "individual":
        args.filter_type = "dual_random_walk"
    if args.graph_type == "dynamic":
        args.filter_type = "dual_random_walk"
    if args.graph_type == "combined":
        args.filter_type = "laplacian"

    return args
