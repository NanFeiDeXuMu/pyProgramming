"""Post-selection stress test on internal holdout identities, never final training data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import json
import joblib
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from assignment_model import make_pipeline, ImageClassifier, strip_training_state, flush_tiny_training_values
from run_experiments import data_and_splits, CACHE, ROOT

def main():
    x,y,v,z,train,clean,dev = data_and_splits()
    selected,_=train_test_split(clean,train_size=1000,stratify=y[clean],random_state=771)
    original=x[selected].copy()
    specifications=[('patch512','patch512_robust_ratio2.0_alpha5.0_mlp384'),
                    ('patch512','patch512_robust2_ratio2.0_alpha5.0_mlp384'),
                    ('patch512_p8','patch512_p8_robust_ratio2.0_alpha5.0_mlp384'),
                    ('patch1024_smooth0.6','patch1024_smooth0.6_robust_ratio1.0_alpha5.0_mlp384')]
    members=[]
    for tag,name in specifications:
        saved=joblib.load(CACHE/(name+'.joblib'))
        flush_tiny_training_values(saved['model'])
        strip_training_state(saved['model'])
        members.append(make_pipeline(joblib.load(CACHE/(tag+'_bank.joblib')),
                                      saved['model'],scaler=saved['scaler']))
    ensemble=ImageClassifier(members,horizontal_tta=True)
    baseline=joblib.load(CACHE/'hog_color_ridge1000.joblib')
    baseline=make_pipeline([],baseline['model'],scaler=baseline['scaler'])
    rng=np.random.default_rng(771)
    tests={'clean':original,
           'gaussian_noise_sigma35':np.clip(original.astype(np.float32)+rng.normal(0,35,original.shape),0,255).astype(np.uint8),
           'blur_sigma1.5':gaussian_filter(original.astype(np.float32),sigma=(0,1.5,1.5,0)).astype(np.uint8),
           'unseen_pixelation_4x':np.repeat(np.repeat(original.astype(np.float32).reshape(-1,8,4,8,4,3).mean(axis=(2,4)),4,axis=1),4,axis=2).astype(np.uint8),
           'unseen_grayscale':np.repeat(np.mean(original,axis=3,keepdims=True),3,axis=3).astype(np.uint8)}
    rows=[]
    for name,images in tests.items():
        with threadpool_limits(limits=1):
            a=baseline.predict(images)
            b=ensemble.predict(images)
        row=dict(condition=name,n=len(images),baseline_accuracy=float(np.mean(a==y[selected])),
                 selected_ensemble_accuracy=float(np.mean(b==y[selected])))
        rows.append(row)
        print(json.dumps(row),flush=True)
    result=dict(note='Post-selection diagnostic only. Development models trained on 45000 images; these 1000 original identities were held out. These are not the final 50000-image-fit models or hidden-test scores.',
                results=rows)
    (ROOT/'robustness_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
