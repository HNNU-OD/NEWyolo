from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import math
import os

PROJECT = Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR'] = str(PROJECT / '.ultralytics')
os.environ['YOLO_AUTOINSTALL'] = 'False'
os.environ['MPLBACKEND'] = 'Agg'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='Train on the entire dataset instead of the smoke subset.')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--model', choices=['yolov8n.pt','yolov8s.pt','yolov8m.pt','yolov8l.pt','yolov8x.pt'], default='yolov8n.pt')
    parser.add_argument('--name', default=None)
    args = parser.parse_args()
    import torch
    import ultralytics
    from ultralytics import YOLO, settings
    from ultralytics.data.utils import check_det_dataset, verify_image_label
    expected = Path('D:/newYOLO/ultralytics/ultralytics/__init__.py').resolve()
    assert Path(ultralytics.__file__).resolve() == expected, ultralytics.__file__
    assert torch.cuda.is_available(), 'Expected the verified NVIDIA GPU environment.'
    disabled = {k:False for k in ['sync','wandb','mlflow','clearml','comet','neptune','dvc','tensorboard'] if k in settings}
    settings.update(disabled)
    env = dict(ultralytics=ultralytics.__version__, source=ultralytics.__file__, torch=torch.__version__,
               cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0))
    (PROJECT/'environment.json').write_text(json.dumps(env, indent=2), encoding='utf-8')
    print(env, flush=True)
    data = check_det_dataset(PROJECT/'data_full.yaml', autodownload=False)
    records = json.loads((PROJECT/'image_inventory.json').read_text(encoding='utf-8'))
    root = Path(data['path'])
    # Use the repository's own verifier on every image/label pair, including test data.
    verify_args = [(str(root/'images'/r['image']), str(root/'labels'/(Path(r['image']).stem+'.txt')),
                    '', False, len(data['names']), 0, 0, False) for r in records]
    totals = dict(missing=0, found=0, empty=0, corrupt=0)
    messages = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for item in pool.map(verify_image_label, verify_args):
            for key,value in zip(totals, item[5:9]):
                totals[key] += int(value)
            if item[9]:
                messages.append(item[9])
    (PROJECT/'ultralytics_verification.json').write_text(json.dumps(dict(totals=totals,messages=messages), indent=2), encoding='utf-8')
    print('Official dataset verification:', totals, flush=True)
    assert totals['corrupt'] == totals['missing'] == 0
    weight_dir = PROJECT/'weights'
    weight_dir.mkdir(exist_ok=True)
    model = YOLO(weight_dir/args.model)
    metrics = model.train(data=str(PROJECT/('data_full.yaml' if args.full else 'smoke.yaml')),
                          epochs=args.epochs, imgsz=640, batch=args.batch, device=0, workers=0,
                          amp=False, cache=False, plots=True, save=True, seed=20260928,
                          project=str(PROJECT/'runs'), name=args.name or ('full' if args.full else 'smoke'),
                          exist_ok=False, val=True, close_mosaic=0)
    run_dir = Path(model.trainer.save_dir)
    best = run_dir/'weights/best.pt'
    assert best.is_file() and (run_dir/'weights/last.pt').is_file()
    trained = YOLO(best)
    assert len(trained.names) == len(data['names']) and all(n.isascii() for n in trained.names.values())
    sample_paths = (PROJECT/'smoke_val.txt').read_text().splitlines()[:4]
    results = trained.predict(source=sample_paths, imgsz=640, device=0, conf=0.25,
                              save=True, project=str(PROJECT/'runs'), name=Path(args.model).stem+'_predict', exist_ok=False)
    prediction_summary = []
    for result in results:
        assert result.orig_img is not None
        assert torch.isfinite(result.boxes.data).all()
        prediction_summary.append(dict(image=result.path, detections=len(result.boxes), save_dir=result.save_dir))
    stats = {k:float(v) for k,v in metrics.results_dict.items()}
    assert all(math.isfinite(v) for v in stats.values())
    summary = dict(status='passed', purpose='pipeline smoke test, not an accuracy evaluation',
                   full_dataset=args.full, epochs=args.epochs, imgsz=640, batch=args.batch,
                   run_dir=str(run_dir), best=str(best), classes=trained.names,
                   model=args.model, metrics=stats, predictions=prediction_summary, environment=env,
                   per_class=[{k:(v.item() if hasattr(v,'item') else v) for k,v in row.items()} for row in metrics.summary()])
    summary_name = 'run_summary.json' if args.model == 'yolov8n.pt' else 'run_summary_'+Path(args.model).stem+'.json'
    (PROJECT/summary_name).write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (run_dir/'run_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('TRAIN / VALIDATE / SAVE / RELOAD / PREDICT: PASSED', flush=True)


if __name__ == '__main__':
    main()
