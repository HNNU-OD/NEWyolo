"""Portable YOLOv8 entry point; keep this file in newYOLO/insulator_project."""
import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'], default='yolov8m.pt')
    parser.add_argument('--name', default=None, help='Optional run directory name.')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='0', help='CUDA device index, or cpu')
    parser.add_argument('--check-only', action='store_true', help='Check all paths and labels without training.')
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    root = project.parent
    repo = root / 'ultralytics'
    dataset = root / 'datasets' / 'insulator_merged_yolo'
    weights = project / 'weights' / args.model
    if not (repo / 'ultralytics' / '__init__.py').is_file():
        raise FileNotFoundError(f'Missing local source repository: {repo}')
    if not weights.is_file():
        raise FileNotFoundError(f'Upload the initial pretrained weight file: {weights}')

    import yaml
    config = yaml.safe_load((dataset / 'data_yolov8.yaml').read_text(encoding='utf-8-sig'))
    assert config['nc'] == len(config['names']) == 5, 'Expected the current five-class dataset.'
    assert set(config['names']) == set(range(5))
    config['path'] = dataset.as_posix()
    totals = {}
    seen = set()
    for split in ['train', 'val', 'test']:
        listing = dataset / config[split]
        entries = listing.read_text(encoding='utf-8-sig').splitlines()
        totals[split] = len(entries)
        for entry in entries:
            image = (listing.parent / entry).resolve()
            if not image.is_file():
                raise FileNotFoundError(image)
            assert image not in seen, f'Duplicate split entry: {image}'
            seen.add(image)
            label = dataset / 'labels' / (image.stem + '.txt')
            if not label.is_file():
                raise FileNotFoundError(label)
            for row in label.read_text(encoding='utf-8-sig').splitlines():
                if row.strip():
                    fields = row.split()
                    assert len(fields) == 5 and int(fields[0]) in config['names'], label
    runtime_yaml = project / 'data_server.yaml'
    runtime_yaml.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    print(f'Dataset: {dataset}\nSplit images: {totals}\nClasses: {config["names"]}', flush=True)
    if args.check_only:
        print('Path, pairing and class checks passed. GPU/dependencies/training have not been tested by this check.')
        return

    sys.path.insert(0, str(repo))
    config_dir = project / '.server_config'
    config_dir.mkdir(exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(config_dir)
    os.environ['YOLO_AUTOINSTALL'] = 'False'
    os.environ['MPLBACKEND'] = 'Agg'
    import torch
    import ultralytics
    from ultralytics import YOLO, settings
    assert Path(ultralytics.__file__).resolve() == (repo/'ultralytics/__init__.py').resolve()
    if args.device != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Install a GPU-enabled PyTorch build for the server, or use --device cpu.')
    settings.update({k:False for k in ['sync','wandb','mlflow','clearml','comet','neptune','dvc','tensorboard'] if k in settings})
    print(f'Ultralytics {ultralytics.__version__}; PyTorch {torch.__version__}; source {ultralytics.__file__}', flush=True)
    model = YOLO(str(weights))
    model.train(data=str(runtime_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                workers=args.workers, device=args.device, amp=False,
                project=str(project/'runs'), name=args.name or f'server_{Path(args.model).stem}', exist_ok=False,
                cache=False, plots=True, seed=20260928, close_mosaic=0)


if __name__ == '__main__':
    main()
