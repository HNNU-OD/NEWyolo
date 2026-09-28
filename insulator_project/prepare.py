from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageOps, ImageChops, ImageStat
import hashlib
import io
import itertools
import json
import math
import random
import re
import shutil
import yaml

SOURCE = Path('D:/traindata/insulator_merged_yolo')
PROJECT = Path('D:/newYOLO/insulator_project')
DATA = Path('D:/newYOLO/datasets/insulator_merged_yolo')
PROJECT.mkdir(parents=True, exist_ok=True)
if not DATA.exists():
    print('Copying dataset into D:/newYOLO/datasets ...', flush=True)
    shutil.copytree(SOURCE, DATA)
cfg = yaml.safe_load((DATA / 'data_yolov8.yaml').read_text(encoding='utf-8-sig'))
cfg['path'] = DATA.as_posix()
for name in ['data.yaml', 'data_yolov8.yaml']:
    (DATA / name).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
splits = {}
errors = []
for split in ['train', 'val', 'test']:
    entries = (DATA / cfg[split]).read_text(encoding='utf-8-sig').splitlines()
    for line in entries:
        p = DATA / line
        if not p.is_file():
            errors.append(['missing_image', line])
        if p.name in splits:
            errors.append(['repeated_split_entry', p.name])
        splits[p.name] = split


def inspect(p):
    issues = []
    raw = p.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            size = im.size
            mode = im.mode
            orientation = im.getexif().get(274, 1)
        if raw[-2:] != b'\xff\xd9':
            issues.append(['jpeg_end_marker', p.name])
    except Exception as exc:
        return {'image': p.name, 'issues': [['image_decode', p.name, str(exc)]]}
    lp = DATA / 'labels' / (p.stem + '.txt')
    boxes = []
    if not lp.exists():
        issues.append(['missing_label', p.name])
    else:
        for number, line in enumerate(lp.read_text(encoding='utf-8-sig').splitlines(), 1):
            if not line.strip():
                continue
            try:
                values = list(map(float, line.split()))
                assert len(values) == 5 and all(math.isfinite(v) for v in values)
                c, x, y, w, h = values
                assert c.is_integer() and int(c) in cfg['names']
                assert 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
                assert min(x-w/2, y-h/2) >= -1e-5 and max(x+w/2, y+h/2) <= 1+1e-5
                boxes.append([int(c), x, y, w, h])
            except (ValueError, AssertionError):
                issues.append(['invalid_label', lp.name, number, line])
    if len({tuple(b) for b in boxes}) != len(boxes):
        issues.append(['duplicate_boxes', lp.name])
    if p.name not in splits:
        issues.append(['unassigned_image', p.name])
    return dict(image=p.name, split=splits.get(p.name), size=size, mode=mode,
                exif_orientation=orientation, sha256=digest, boxes=boxes, issues=issues)


images = sorted((DATA / 'images').glob('*.jpg'))
records = []
with ThreadPoolExecutor(max_workers=4) as pool:
    for n, record in enumerate(pool.map(inspect, images), 1):
        records.append(record)
        errors.extend(record['issues'])
        if n % 250 == 0:
            print(f'Checked {n}/{len(images)} images', flush=True)
stems = {p.stem for p in images}
for p in (DATA / 'labels').glob('*.txt'):
    if p.stem not in stems:
        errors.append(['orphan_label', p.name])
hashes = defaultdict(list)
counts = Counter()
for r in records:
    if 'sha256' in r:
        hashes[r['sha256']].append(r['image'])
        counts.update(b[0] for b in r['boxes'])
report = dict(images=len(images), labels=len(list((DATA/'labels').glob('*.txt'))),
              boxes=sum(counts.values()), class_boxes=dict(sorted(counts.items())),
              split_images=dict(Counter(splits.values())), errors=errors,
              empty_labels=[r['image'] for r in records if 'boxes' in r and not r['boxes']],
              exact_duplicate_groups=[v for v in hashes.values() if len(v)>1],
              non_default_exif=[r['image'] for r in records if r.get('exif_orientation',1)!=1])
# Check known flip variants without interpreting byte-level uniqueness as independence.
groups = defaultdict(list)
for r in records:
    if r['image'].startswith('idd_'):
        groups[re.sub('[dhv]$', '', Path(r['image']).stem)].append(r)
near_duplicates = []
for group in groups.values():
    if len({r['split'] for r in group}) < 2:
        continue
    thumbs = {}
    for r in group:
        with Image.open(DATA/'images'/r['image']) as im:
            thumbs[r['image']] = im.convert('RGB').resize((128,128))
    best = None
    for a,b in itertools.combinations(group,2):
        if a['split'] == b['split']:
            continue
        im = thumbs[a['image']]
        variants = [im, ImageOps.mirror(im), ImageOps.flip(im), im.transpose(Image.Transpose.ROTATE_180)]
        mae = min(sum(ImageStat.Stat(ImageChops.difference(v, thumbs[b['image']])).mean)/3 for v in variants)
        if best is None or mae < best['mae']:
            best = dict(a=a['image'], b=b['image'], mae=mae)
    if best and best['mae'] < 3:
        near_duplicates.append(best)
report['cross_split_similar_idd_groups'] = near_duplicates
(PROJECT/'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
(PROJECT/'image_inventory.json').write_text(json.dumps(records, ensure_ascii=False), encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ['cross_split_similar_idd_groups','empty_labels']}, ensure_ascii=True), flush=True)
print(f'Empty labels: {len(report["empty_labels"])}; similar IDD groups across splits: {len(near_duplicates)}', flush=True)
if errors:
    raise SystemExit('Dataset problems found; inspect audit.json before training.')
# Deliberately small smoke set with every class represented, separate from full data.
rng = random.Random(20260928)
smoke = dict(cfg)
smoke.pop('test', None)
selection = {}
for split, limit in [('train',64), ('val',32)]:
    pool = [r for r in records if r['split']==split]
    rng.shuffle(pool)
    chosen = {}
    for c in range(cfg['nc']):
        candidate = next(r for r in pool if any(b[0]==c for b in r['boxes']))
        chosen[candidate['image']] = candidate
    for r in pool:
        if len(chosen) >= limit:
            break
        chosen[r['image']] = r
    listing = PROJECT / f'smoke_{split}.txt'
    listing.write_text('\n'.join((DATA/'images'/name).as_posix() for name in chosen)+'\n', encoding='utf-8')
    smoke[split] = listing.as_posix()
    selection[split] = {'images':len(chosen), 'class_boxes':dict(Counter(b[0] for r in chosen.values() for b in r['boxes']))}
(PROJECT/'smoke.yaml').write_text(yaml.safe_dump(smoke, sort_keys=False), encoding='utf-8')
(PROJECT/'smoke_selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
shutil.copy2(DATA/'data_yolov8.yaml', PROJECT/'data_full.yaml')
print('Full dataset and smoke configuration prepared.', flush=True)
