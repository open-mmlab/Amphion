import os
import sys
import subprocess
from argparse import ArgumentParser


# Set up directories
work_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
print(f"Work Directory: {work_dir}")

os.environ['WORK_DIR'] = work_dir
os.environ['PYTHONPATH'] = work_dir
os.environ['PYTHONIOENCODING'] = 'UTF-8'

# Build Monotonic Align Module
os.chdir(os.path.join(work_dir, 'modules', 'monotonic_align'))
os.makedirs('monotonic_align', exist_ok=True)
subprocess.run(['python', 'setup.py', 'build_ext', '--inplace'], check=True)
os.chdir(work_dir)

# Parse parameters
parser = ArgumentParser()
parser.add_argument('-c', '--config', help='Experimental Configuration File')
parser.add_argument('-n', '--name', help='Experimental Name')
parser.add_argument('-s', '--stage', type=int, help='Running Stage')
parser.add_argument('--gpu', type=str, help='Visible GPU machines')
parser.add_argument('--model_train_stage', type=str, help='Model Training Stage')
parser.add_argument('--ar_model_ckpt_dir', type=str, help='The stage1 ckpt dir')
parser.add_argument('--infer_expt_dir', type=str, help='The experiment dir')
parser.add_argument('--infer_output_dir', type=str, help='The output dir to save inferred audios')
parser.add_argument('--infer_mode', type=str, help='The inference mode')
parser.add_argument('--infer_test_list_file', type=str, help='The inference test list file')
parser.add_argument('--infer_text', type=str, help='The text to be synthesized from')
parser.add_argument('--infer_text_prompt', type=str, help='The inference text prompt')
parser.add_argument('--infer_audio_prompt', type=str, help='The inference audio prompt')
args = parser.parse_args()

# Check required parameters
if args.stage is None:
    print("Error: Please specify the running stage")
    sys.exit(1)

if args.config is None:
    args.config = os.path.join(work_dir, 'exp_config.json')
print(f"Experimental Configuration File: {args.config}")

if args.gpu is None:
    args.gpu = '0'

# Features Extraction
if args.stage == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cmd = ['python', os.path.join(work_dir, "bins", "tts", "preprocess.py"), '--config', args.config, '--num_workers', '4']
    subprocess.run(cmd, check=True, cwd=work_dir)

# Training
if args.stage ==   2:
    if args.name is None:
        print("Error: Please specify the experiments name")
        sys.exit(1)

    if args.model_train_stage == '2' and args.ar_model_ckpt_dir is None:
        print("Error: Please specify the checkpoint path to the trained model in stage1.")
        sys.exit(1)

    if args.model_train_stage == '1':
        args.ar_model_ckpt_dir = None

    print(f"Experimental Name: {args.name}")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cmd = ['python', os.path.join(work_dir, "bins", "tts", "train.py"), '--config', args.config, '--exp_name', args.name, '--log_level', 'debug', '--train_stage', args.model_train_stage, '--checkpoint_path', args.ar_model_ckpt_dir]
    subprocess.run(cmd, check=True)

# Inference
if args.stage ==   3:
    if args.infer_expt_dir is None:
        print("Error: Please specify the experimental directory. The value is like [Your path to save logs and checkpoints]/[YourExptName]")
        sys.exit(1)

    if args.infer_output_dir is None:
        args.infer_output_dir = os.path.join(args.infer_expt_dir, 'result')

    if args.infer_mode is None:
        print("Error: Please specify the inference mode, e.g., \"batch\", \"single\"")
        sys.exit(1)

    if args.infer_mode == 'batch' and args.infer_test_list_file is None:
        print("Error: Please specify the test list file used in inference when the inference mode is batch")
        sys.exit(1)

    if args.infer_mode == 'single' and args.infer_text is None:
        print("Error: Please specify the text to be synthesized when the inference mode is single")
        sys.exit(1)

    if args.infer_mode == 'single':
        print(f'Text: {args.infer_text}')
        args.infer_test_list_file = None
    elif args.infer_mode == 'batch':
        args.infer_text = ""
        args.infer_text_prompt = ""
        args.infer_audio_prompt = ""

    cmd = ['python', os.path.join(work_dir, "bins", "tts", "inference.py"), '--config', args.config, '--log_level', 'debug', '--acoustics_dir', args.infer_expt_dir, '--output_dir', args.infer_output_dir, '--mode', args.infer_mode, '--text', args.infer_text, '--text_prompt', args.infer_text_prompt, '--audio_prompt', args.infer_audio_prompt, '--test_list_file', args.infer_test_list_file]
    subprocess.run(cmd, check=True)
